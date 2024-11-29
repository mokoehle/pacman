# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DefensiveAgent', second='OffensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [DefensiveAgent(first_index), OffensiveReflexAgent(second_index)]


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
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}
    
class QLearningAgent(CaptureAgent):
    """
    A base class for Q-learning agents
    """

    def __init__(self, index, time_for_computing=.1, alpha=0.2, epsilon=0.05, gamma=0.8):
        super().__init__(index, time_for_computing)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_values = util.Counter()
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if util.flip_coin(self.epsilon):
            return random.choice(actions)
        else:
            return self.compute_action_from_q_values(game_state)

    def compute_action_from_q_values(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if len(actions) == 0:
            return None

        values = [self.get_q_value(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def get_q_value(self, game_state, action):
        return self.q_values[(game_state, action)]

    def update(self, game_state, action, next_state, reward):
        old_q_value = self.get_q_value(game_state, action)
        future_rewards = [self.get_q_value(next_state, a) for a in next_state.get_legal_actions(self.index)]
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max(future_rewards))
        self.q_values[(game_state, action)] = new_q_value

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor
        
class OffensiveQLearningAgent(QLearningAgent):
    """
    A Q-learning agent that seeks food and returns it to its own side
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveQLearningAgent(QLearningAgent):
    """
    A Q-learning agent that defends its side and catches invaders
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

# class OffensiveAgent(CaptureAgent): 

#     def __init__(self, index, epsilon=0.1, alpha=0.5, discount=0.9):
#         super().__init__(index)
#         self.epsilon = epsilon  # Exploration rate
#         self.alpha = alpha      # Learning rate
#         self.discount = discount  # Discount factor
#         self.q_values = {}  # Q-Value dictionary for state-action pairs

#     def register_initial_state(self, game_state):
#         """
#         Initialisiert den Agenten für das Spiel und ruft die Methode
#         von CaptureAgent auf.
#         """
#         CaptureAgent.register_initial_state(self, game_state)

#     def choose_action(self, game_state):
#         """
#         Wählt eine Aktion basierend auf einer Epsilon-Greedy-Strategie aus.
#         """
#         actions = game_state.get_legal_actions(self.index)
#         if not actions:
#             return Directions.STOP

#         if random.random() < self.epsilon:
#             # Zufällige Aktion zur Exploration
#             chosen_action = random.choice(actions)
#         else:
#             # Beste Aktion basierend auf Q-Werten
#             values = [self.get_q_value(game_state, action) for action in actions]
#             max_value = max(values)
#             best_actions = [action for action, value in zip(actions, values) if value == max_value]
#             chosen_action = random.choice(best_actions)

#         # Q-Learning-Update
#         self.update_q_value(game_state, chosen_action)

#         return chosen_action

#     def get_q_value(self, game_state, action):
#         state = self.get_state_representation(game_state)
#         return self.q_values.get((state, action), 0.0)

#     def update_q_value(self, game_state, action):
#         state = self.get_state_representation(game_state)
#         reward = self.get_reward(game_state)
#         next_state = self.get_successor(game_state, action)
#         next_actions = next_state.get_legal_actions(self.index)
#         if next_actions:
#             future_rewards = max([self.get_q_value(next_state, next_action) for next_action in next_actions])
#         else:
#             future_rewards = 0.0

#         # Q-Learning Update Regel
#         sample = reward + self.discount * future_rewards
#         self.q_values[(state, action)] = (1 - self.alpha) * self.get_q_value(game_state, action) + self.alpha * sample

#     def get_reward(self, game_state):
#         reward = 0
#         my_state = game_state.get_agent_state(self.index)

#         if my_state.is_pacman: 
#             if game_state.get_agent_state(self.index).num_carrying > 0: 
#                 reward += 10
#             if my_state.scared_timer > 0: 
#                 reward -= 500
        
#         else: 
#             reward -= 1 # minor punishment for every step not collecting 

#         return reward

#     def get_state_representation(self, game_state):
#         my_pos = game_state.get_agent_position(self.index)
#         food_list = self.get_food(game_state).as_list()
#         return (my_pos, tuple(food_list))
    
#     def evaluate(self, game_state, action): 
#         features = self.get_features_offensive(game_state, action) 
#         weights = self.get_weights_offensive(game_state, action)
#         return features * weights
    
#     def get_successor(self, game_state, action):
#         successor = game_state.generate_successor(self.index, action)
#         pos = successor.get_agent_state(self.index).get_position()
#         if pos != nearest_point(pos):
#             return successor.generate_successor(self.index, action)
#         else:
#             return successor
        
#     def get_features_offensive(self, game_state, action): 
#         features = util.Counter()
#         successor = self.get_successor(game_state, action) 
#         my_pos = successor.get_agent_state(self.index).get_position()

#         food_list = self.get_food(successor).as_list()
#         features['successor_score'] = -len(food_list)

#         if len(food_list) > 0: 
#             min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
#             features['distance_to_food'] = min_distance

#         return features
    
#     def get_weights_offensive(self, game_state, action): 
#         return {
#             'successor_score': 100, 
#             'distance_to_food': -10
#         }
    
class DefensiveAgent(CaptureAgent): 

    def register_initial_state(self, game_state): 
        # Initialize the agent for the game
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state): 
        # Get legal actions for the agent
        actions = game_state.get_legal_actions(self.index) 
        if not actions: 
            return Directions.STOP
        
        # Decide whether to attack or defend
        if self.should_attack(game_state): 
            values = [self.evaluate_offensive(game_state, action) for action in actions]
        else: 
            values = [self.evaluate_defensive(game_state, action) for action in actions]
        
        if len(values) == 0: 
            return Directions.STOP
        
        # Choose the best action based on evaluation
        max_value = max(values) 
        best_actions = [action for action, value in zip(actions, values) if value == max_value]

        if len(best_actions) == 0: 
            return Directions.STOP
        
        return random.choice(best_actions) 

    def should_attack(self, game_state):
        # Determine if the agent should attack based on the presence of invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        if len(invaders) == 0:
            return True
        return False
    
    def get_successor(self, game_state, action): 
        # Generate the successor state after taking an action
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos): 
            return successor.generate_successor(self.index, action)
        else: 
            return successor

    def evaluate_offensive(self, game_state, action): 
        # Evaluate the action based on offensive features and weights
        features = self.get_features_offensive(game_state, action) 
        weights = self.get_weights_offensive(game_state, action) 
        return features * weights
    
    def evaluate_defensive(self, game_state, action): 
        # Evaluate the action based on defensive features and weights
        features = self.get_features_defensive(game_state, action) 
        weights = self.get_weights_defensive(game_state, action) 
        return features * weights
    
    def evaluate(self, game_state, action): 
        # General evaluation function
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights
    
    def get_features(self, game_state, action): 
        """
        Extracts features for the given game state and action.

        Args:
            game_state (GameState): The current state of the game.
            action (str): The action to be evaluated.

        Returns:
            util.Counter: A counter of features and their corresponding values.

        Features:
            - 'on_defense': 1 if the agent is on defense, 0 if it is a pacman.
            - 'num_invaders': The number of invaders (enemy pacmen) in the agent's territory.
            - 'invader_distance': The distance to the closest invader.
            - 'distance_to_food': The distance to the closest food if there are no invaders.
            - 'stop': 1 if the action is to stop, 0 otherwise.
            - 'reverse': 1 if the action is to reverse direction, 0 otherwise.
        """
        # Extract features for the given state and action
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0: 
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if len(invaders) == 0: 
            food_list = self.get_food(successor).as_list()
            if len(food_list) > 0: 
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance

        if action == Directions.STOP: 
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features
    
    def get_features_offensive(self, game_state, action):
        # Extract offensive features for the given state and action
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()

        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        return features

    def get_features_defensive(self, game_state, action):
        # Extract defensive features for the given state and action
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights_offensive(self, game_state, action):
        # Define weights for offensive features
        return {
            'successor_score': 100,
            'distance_to_food': -1
        }

    def get_weights_defensive(self, game_state, action):
        # Define weights for defensive features
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2
        }

    def get_weights(self, game_state, action): 
        # General weights for features
        return {
            'num_invaders': -1000, 
            'on_defense': 100,
            'invader_distance': -10, 
            'distance_to_food': -5,
            'stop': -100, 
            'reverse': -2
        }
