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
# A stronger reflex-based team for the capture-the-flag contest.

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions.
    Adds a few utility methods and a shared eval() implementation.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.mid_x = None  # dividing line between sides

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)

        # Compute the x coordinate of the border between teams to define "home"
        layout = game_state.data.layout
        width = layout.width
        if game_state.is_on_red_team(self.index):
            self.mid_x = (width // 2) - 1
        else:
            self.mid_x = width // 2

        # Pre-compute legal "home" border positions (used for retreating / patrolling)
        self.border_positions = []
        for y in range(1, layout.height - 1):
            if not game_state.has_wall(self.mid_x, y):
                self.border_positions.append((self.mid_x, y))

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        actions = [a for a in actions if a != Directions.STOP] or [Directions.STOP]

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Tie-breaking with a bit of randomness
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            successor = successor.generate_successor(self.index, action)
        return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    # Default versions – subclasses override these
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}

    # ---- Convenience helpers ----

    def get_my_successor_pos(self, game_state, action):
        succ = self.get_successor(game_state, action)
        return succ.get_agent_state(self.index).get_position()

    def get_visible_enemies(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [e for e in enemies if e.get_position() is not None]

    def is_on_home_side(self, pos):
        if pos is None:
            return True
        x, y = pos
        if self.red:
            return x <= self.mid_x
        else:
            return x >= self.mid_x


############################
# Offensive Reflex Agent   #
############################

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A stronger offensive agent:

    - Hunts nearby food and capsules
    - Avoids visible enemy ghosts
    - Returns home when carrying a lot of food or in danger
    - Uses scared-timer information to decide to chase or run
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.return_food_threshold = 5  # go home when carrying this much
        self.safe_distance_from_ghost = 4

    def get_features(self, game_state, action):
        features = util.Counter()

        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # --- Basic score ---
        features['successor_score'] = self.get_score(successor)

        # --- Stop & reverse penalties ---
        if action == Directions.STOP:
            features['stop'] = 1

        current = game_state.get_agent_state(self.index)
        current_dir = current.configuration.direction
        rev = Directions.REVERSE[current_dir]
        if action == rev:
            features['reverse'] = 1

        # --- Food related ---
        food_list = self.get_food(successor).as_list()
        if len(food_list) > 0:
            min_food_dist = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_food_dist
        else:
            features['distance_to_food'] = 0

        # Bonus for actually eating food this turn
        prev_food = self.get_food(game_state).as_list()
        features['ate_food'] = 1 if len(prev_food) > len(food_list) else 0

        # --- Capsules ---
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            min_capsule_dist = min(self.get_maze_distance(my_pos, c) for c in capsules)
            features['distance_to_capsule'] = min_capsule_dist

        # --- Enemies ---
        enemies = self.get_visible_enemies(successor)
        ghosts = [e for e in enemies if not e.is_pacman]
        ghost_dists = []
        scared_ghost_dists = []

        for g in ghosts:
            dist = self.get_maze_distance(my_pos, g.get_position())
            if g.scared_timer > 0:
                scared_ghost_dists.append(dist)
            else:
                ghost_dists.append(dist)

        if ghost_dists:
            min_ghost_dist = min(ghost_dists)
            features['closest_ghost'] = min_ghost_dist

            # Emergency: if a ghost is really close, strongly encourage running away
            if min_ghost_dist <= 1:
                features['danger'] = 1

        if scared_ghost_dists:
            features['closest_scared_ghost'] = min(scared_ghost_dists)

        # --- Carrying food & returning home ---
        carried = my_state.num_carrying
        features['carrying'] = carried

        # If carrying a lot of food or the game is almost over, head home.
        time_left = successor.data.timeleft
        go_home = (
            carried >= self.return_food_threshold or
            (time_left is not None and time_left < 100)
        )

        # Also go home if we see a ghost very close
        if ghost_dists and min(ghost_dists) <= 2:
            go_home = True

        if go_home:
            # Distance from current position to nearest border point on our side
            if self.border_positions:
                home_dists = [self.get_maze_distance(my_pos, p) for p in self.border_positions]
                features['distance_to_home'] = min(home_dists)

        return features

    def get_weights(self, game_state, action):
        """
        Larger magnitude = more important feature.
        """
        return {
            'successor_score': 200.0,

            # Food / capsules
            'distance_to_food': -3.0,
            'ate_food': 20.0,
            'distance_to_capsule': -2.0,

            # Ghost avoidance / chasing
            'closest_ghost': 4.0,             # prefer larger distance (positive weight)
            'closest_scared_ghost': -3.0,     # move TOWARDS scared ghosts
            'danger': -1000.0,                # never step next to an unsuper scared ghost if possible

            # Returning home
            'distance_to_home': -5.0,
            'carrying': 10.0,                 # value carrying; combined with distance_to_home drives retreat

            # Movement smoothness
            'stop': -100.0,
            'reverse': -2.0,
        }


############################
# Defensive Reflex Agent   #
############################

def get_defensive_features(agent, game_state, action):
    """
    Common defensive feature extractor used by both agents when needed.
    """
    features = util.Counter()
    successor = agent.get_successor(game_state, action)

    my_state = successor.get_agent_state(agent.index)
    my_pos = my_state.get_position()

    # Stay on defense (on home side) when possible
    features['on_defense'] = 1
    if my_state.is_pacman:
        features['on_defense'] = 0

    # Visible enemies
    enemies = [successor.get_agent_state(i) for i in agent.get_opponents(successor)]
    invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

    features['num_invaders'] = len(invaders)

    if invaders:
        dists = [agent.get_maze_distance(my_pos, i.get_position()) for i in invaders]
        features['invader_distance'] = min(dists)

        # If scared, we *don't* want to get closer to invaders
        if my_state.scared_timer > 0:
            features['run_from_invader'] = -features['invader_distance']
    else:
        # Patrol midline when no invaders
        if hasattr(agent, 'patrol_points') and agent.patrol_points:
            patrol_dists = [agent.get_maze_distance(my_pos, p) for p in agent.patrol_points]
            features['patrol_distance'] = min(patrol_dists)

    # Avoid stopping or reversing
    if action == Directions.STOP:
        features['stop'] = 1
    cur_dir = game_state.get_agent_state(agent.index).configuration.direction
    rev = Directions.REVERSE[cur_dir]
    if action == rev:
        features['reverse'] = 1

    # Bonus: respond to recently eaten food on our side
    if hasattr(agent, 'last_defended_food') and agent.last_defended_food:
        # If some defensive food disappeared since last observation, that set is stored in agent.last_eaten_positions
        if hasattr(agent, 'last_eaten_positions') and agent.last_eaten_positions:
            dists = [agent.get_maze_distance(my_pos, p) for p in agent.last_eaten_positions]
            features['dist_to_eaten_food'] = min(dists)

    return features


def get_defensive_weights(agent, game_state, action):
    return {
        'num_invaders': -1000.0,
        'invader_distance': -5.0,
        'run_from_invader': 5.0,     # when scared, increase distance
        'on_defense': 100.0,
        'patrol_distance': -2.0,
        'dist_to_eaten_food': -4.0,
        'stop': -100.0,
        'reverse': -2.0,
    }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A stronger defensive agent:

    - Chases visible invaders
    - Patrols the central border when there are no invaders
    - Responds to defensive food that has just been eaten
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_points = []
        self.last_defended_food = None
        self.last_eaten_positions = []

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Initialize patrol points along the central border
        self.patrol_points = list(self.border_positions)

        # Track defensive food
        self.last_defended_food = self.get_food_you_are_defending(game_state).as_list()
        self.last_eaten_positions = []

    def get_features(self, game_state, action):
        # Update info about eaten defensive food
        current_defended_food = self.get_food_you_are_defending(game_state).as_list()
        if self.last_defended_food is not None:
            eaten = set(self.last_defended_food) - set(current_defended_food)
            if eaten:
                self.last_eaten_positions = list(eaten)
        self.last_defended_food = current_defended_food

        return get_defensive_features(self, game_state, action)

    def get_weights(self, game_state, action):
        return get_defensive_weights(self, game_state, action)
