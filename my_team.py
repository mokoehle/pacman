import random
import contest.util as util
import heapq

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from game import Actions  # Import Actions here
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DefensiveAgentFinal', second='PowerPelletAgent', num_training=0):

    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None

            for action in best_actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor


class DefensiveAgentFinal(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free and also collects food.
    """
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.a_star_search(my_pos, a.get_position(), game_state) for a in invaders]
            features['invader_distance'] = min(dists)

        # Add a feature for stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # Add a feature for reversing direction
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'on_defense': 100,
            'num_invaders': -1000,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2
        }

    def a_star_search(self, start, goal, game_state):
        """
        A* search algorithm to find the shortest path from start to goal.
        """
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                return g_score[current]

            for neighbor in self.get_neighbors(current, game_state):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return float('inf')

    def heuristic(self, pos, goal):
        """
        Heuristic function for A* search.
        """
        return self.get_maze_distance(pos, goal)

    def get_neighbors(self, pos, game_state):
        """
        Get neighbors of the current position.
        """
        neighbors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = pos
            dx, dy = Actions.direction_to_vector(direction)
            next_x, next_y = int(x + dx), int(y + dy)
            if not game_state.has_wall(next_x, next_y):
                neighbors.append((next_x, next_y))
        return neighbors


class PowerPelletAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks power pellets (capsules), returns to its own side,
    and eats enemy Pacmen when on its own side.
    """
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Compute distance to the nearest capsule on the opponent's side
        capsule_list = self.get_capsules(successor)
        opponent_side_capsules = [capsule for capsule in capsule_list if self.is_on_opponent_side(capsule)]
        if len(opponent_side_capsules) > 0:
            min_capsule_distance = min([self.get_maze_distance(my_pos, capsule) for capsule in opponent_side_capsules])
            features['distance_to_capsule'] = min_capsule_distance

        # Compute distance to the nearest food on the opponent's side
        food_list = self.get_food(successor).as_list()
        opponent_side_food = [food for food in food_list if self.is_on_opponent_side(food)]
        if len(opponent_side_food) > 0:
            min_food_distance = min([self.get_maze_distance(my_pos, food) for food in opponent_side_food])
            features['distance_to_food'] = min_food_distance

        # Compute distance to the nearest ghost if the agent is a Pacman
        if my_state.is_pacman:
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
            if len(ghosts) > 0:
                ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
                features['distance_to_ghost'] = min(ghost_distances)

        # Compute distance to the nearest enemy Pacman if the agent is a ghost
        if not my_state.is_pacman:
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            enemy_pacmen = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            if len(enemy_pacmen) > 0:
                enemy_pacmen_distances = [self.get_maze_distance(my_pos, pacman.get_position()) for pacman in enemy_pacmen]
                features['distance_to_enemy_pacman'] = min(enemy_pacmen_distances)

        # Add a feature for stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # Add a feature for reversing direction
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        weights = {
            'distance_to_capsule': -10,  # Increase priority for capsules
            'distance_to_food': -5,  # Increase priority for food
            'distance_to_ghost': 1,  # Negative weight to avoid ghosts
            'distance_to_enemy_pacman': -10,  # Increase priority for eating enemy Pacmen
            'stop': -100,  # Strongly discourage stopping
            'reverse': -2  # Discourage reversing direction
        }

        # Increase the weight for returning to own side if carrying a capsule
        my_state = game_state.get_agent_state(self.index)
        if my_state.num_carrying > 0:
            weights['distance_to_capsule'] = 0  # Stop prioritizing capsules
            weights['distance_to_border'] = -100  # Prioritize returning to own side

        # If the agent has eaten a capsule, prioritize attacking ghosts
        if my_state.scared_timer > 0:
            weights['distance_to_ghost'] = -10  # Prioritize attacking ghosts

        return weights

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def is_on_opponent_side(self, pos):
        """
        Check if a position is on the opponent's side.
        """
        mid_x = self.get_border_x()
        if self.red:
            return pos[0] > mid_x
        else:
            return pos[0] < mid_x

    def get_border_x(self):
        """
        Get the x-coordinate of the border closest to the agent's starting side.
        """
        mid_x = self.get_current_observation().data.layout.width // 2
        if self.red:
            mid_x -= 1
        else:
            mid_x += 1
        return mid_x

    def get_distance_to_border(self, my_pos, game_state):
        """
        Get the distance to the border closest to the agent's starting side.
        """
        mid_x = self.get_border_x()
        border_positions = [(mid_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(mid_x, y)]
        return min([self.get_maze_distance(my_pos, border) for border in border_positions])