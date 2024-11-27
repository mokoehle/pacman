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
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
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
    return [OffensiveAgent(first_index), DefensiveAgent(second_index)]


##########
# Agents #
##########
class OffensiveAgent(CaptureAgent): 

    def __init__(self, index, epsilon=0.1, alpha=0.5, discount=0.9):
        super().__init__(index)
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.discount = discount  # Discount factor
        self.q_values = {}  # Q-Value dictionary for state-action pairs

    def register_initial_state(self, game_state):
        """
        Initialisiert den Agenten für das Spiel und ruft die Methode
        von CaptureAgent auf.
        """
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Wählt eine Aktion basierend auf einer Epsilon-Greedy-Strategie aus.
        """
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP

        if random.random() < self.epsilon:
            # Zufällige Aktion zur Exploration
            chosen_action = random.choice(actions)
        else:
            # Beste Aktion basierend auf Q-Werten
            values = [self.get_q_value(game_state, action) for action in actions]
            max_value = max(values)
            best_actions = [action for action, value in zip(actions, values) if value == max_value]
            chosen_action = random.choice(best_actions)

        # Q-Learning-Update
        self.update_q_value(game_state, chosen_action)

        return chosen_action

    def get_q_value(self, game_state, action):
        state = self.get_state_representation(game_state)
        return self.q_values.get((state, action), 0.0)

    def update_q_value(self, game_state, action):
        state = self.get_state_representation(game_state)
        reward = self.get_reward(game_state)
        next_state = self.get_successor(game_state, action)
        next_actions = next_state.get_legal_actions(self.index)
        if next_actions:
            future_rewards = max([self.get_q_value(next_state, next_action) for next_action in next_actions])
        else:
            future_rewards = 0.0

        # Q-Learning Update Regel
        sample = reward + self.discount * future_rewards
        self.q_values[(state, action)] = (1 - self.alpha) * self.get_q_value(game_state, action) + self.alpha * sample

    def get_reward(self, game_state):
        reward = 0
        my_state = game_state.get_agent_state(self.index)

        if my_state.is_pacman: 
            if game_state.get_agent_state(self.index).num_carrying > 0: 
                reward += 10
            if my_state.scared_timer > 0: 
                reward -= 500
        
        else: 
            reward -= 1 # minor punishment for every step not collecting 

        return reward

    def get_state_representation(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        return (my_pos, tuple(food_list))
    
    def evaluate(self, game_state, action): 
        features = self.get_features_offensive(game_state, action) 
        weights = self.get_weights_offensive(game_state, action)
        return features * weights
    
    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor
        
    def get_features_offensive(self, game_state, action): 
        features = util.Counter()
        successor = self.get_successor(game_state, action) 
        my_pos = successor.get_agent_state(self.index).get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        if len(food_list) > 0: 
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        if not self.is_on_opponents_side(my_pos): 
            features['stay_on_own_side'] = 1

        return features
    
    def get_weights_offensive(self, game_state, action): 
        return {
            'successor_score': 100, 
            'distance_to_food': -10, 
            'stay_on_own_side': -50
        }
    
    def is_on_opponents_side(self, position): 
        mid_x = self.get_food_you_are_defending().width // 2
        if self.red: 
            return position[0] >= mid_x
        else: 
            return position[0] < mid_x
    
class DefensiveAgent(CaptureAgent): 

    def should_attack(self, game_state):
        # Überprüft, ob es sinnvoll ist, die Seite zu wechseln
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        # Wenn keine Eindringlinge auf der eigenen Seite sind, sollte der Agent offensiv agieren
        if len(invaders) == 0:
            return True
        return False
    
    def get_successor(self, game_state, action): 
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos): 
            return successor.generate_successor(self.index, action)
        else: 
            return successor
        
    def register_initial_state(self, game_state): 
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state): 
        actions = game_state.get_legal_actions(self.index) 
        if not actions: 
            return Directions.STOP
        
        if self.should_attack(game_state): 
            # actions = game_state.get_legal_actions(self.index)
            values = [self.evaluate_offensive(game_state, action) for action in actions]
        else: 
            # actions = game_state.get_legal_actions(self.index)
            values = [self.evaluate_defensive(game_state, action) for action in actions]
        
        if len(values) == 0: 
            return Directions.STOP
        
        max_value = max(values) 
        best_actions = [action for action, value in zip(actions, values) if value == max_value]

        if len(best_actions) == 0: 
            return Directions.STOP
        
        return random.choice(best_actions) 
    
    def evaluate_offensive(self, game_state, action): 
        features = self.get_features_offensive(game_state, action) 
        weights = self.get_weights_offensive(game_state, action) 
        return features * weights
    
    def evaluate_defensive(self, game_state, action): 
        features = self.get_features_defensive(game_state, action) 
        weights = self.get_weights_defensive(game_state, action) 
        return features * weights
    
    def evaluate(self, game_state, action): 
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights
    
    def get_features(self, game_state, action): 
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # defensive behavior
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0

        # computes the distance to attackers, which can be seen
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0: 
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # switch to offense 
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
        """
        Bestimmt die Features für offensives Verhalten.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()

        # Anzahl verbleibender Futterpunkte auf der gegnerischen Seite minimieren
        features['successor_score'] = -len(food_list)

        # Abstand zum nächsten Futter berechnen
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        return features

    def get_weights_offensive(self, game_state, action):
        """
        Bestimmt die Gewichtung der Features für offensives Verhalten.
        """
        return {
            'successor_score': 100,
            'distance_to_food': -1
        }
    def get_features_defensive(self, game_state, action):
        """
        Bestimmt die Features für defensives Verhalten.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Berechnet, ob der Agent auf Verteidigung ist (1) oder nicht (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Berechnet den Abstand zu Eindringlingen, die gesehen werden können
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

    def get_weights_defensive(self, game_state, action):
        """
        Bestimmt die Gewichtung der Features für defensives Verhalten.
        """
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2
        }
    def get_weights(self, game_state, action): 
        return {
            'num_invaders': -1000, 
            'on_defense': 100,
            'invader_distance': -10, 
            'distance_to_food': -5,
            'stop': -100, 
            'reverse': -2
        }
