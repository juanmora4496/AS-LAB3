# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).

import random
import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='SmartAgent', second='SmartAgent', num_training=0):
    """
    This function returns a list of two agents that will form the
    team, initialized using firstIndex and secondIndex.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class SmartAgent(CaptureAgent):
    """
    A Hybrid agent that uses a Finite State Machine to switch between
    Offense (Attack/Retreat) and Defense based on the team score.

    Strategy Overview:
    1. High-Level Architecture & Meta-Strategy:
       - Finite State Machine (FSM): Switches between ATTACK, RETREAT, DEFEND.
       - Dynamic Role Allocation: Agent closest to enemy food becomes Attacker.
       - "Lockdown" Protocol: If score lead >= 8, both agents DEFEND.

    2. Offensive Strategies (ATTACK Mode):
       - Greedy Food Collection: Uses A* distance to nearest food.
       - Capsule Prioritization: High priority to eating power capsules.
       - Opportunistic Ghost Hunting: Chases scared ghosts if very close (< 3 units).

    3. Retreat Strategies (RETREAT Mode):
       - Threshold-Based Retreat: Carries 5+ food or time is running out.
       - Safe-Pathing: Linear penalty for positions near ghosts to avoid getting trapped.

    4. Defensive Strategies (DEFEND Mode):
       - Invader Interception: Minimizes distance to visible invaders.
       - Emergency Capsule Protection: Intercepts invaders near capsules (< 5 units).
       - Proactive Capsule Guarding: Guards unguarded capsules when idle.
       - Collaborative Patrol: Patrols midline, spreading out from teammate.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.mode = 'ATTACK' 
        self.patrol_points = []

    def register_initial_state(self, game_state):
        """
        Calculates the home boundary and patrol points for defense.
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        # Calculate defensive patrol points (midline of the map)
        self.patrol_points = []
        map_width = game_state.data.layout.width
        map_height = game_state.data.layout.height
        
        if self.red:
            boundary_x = (map_width // 2) - 1
        else:
            boundary_x = (map_width // 2)

        for y in range(1, map_height - 1):
            if not game_state.has_wall(boundary_x, y):
                self.patrol_points.append((boundary_x, y))

    def choose_action(self, game_state):
        """
        THE BRAIN: Finite State Machine Logic
        """
        # 1. Update Global State Info
        global_score = self.get_score(game_state)
        if self.red:
            current_lead = global_score
        else:
            current_lead = -global_score

        # 2. Identify Teammate and Positions
        team_indices = self.get_team(game_state)
        teammate_index = [i for i in team_indices if i != self.index][0]
        
        my_pos = game_state.get_agent_position(self.index)
        teammate_pos = game_state.get_agent_position(teammate_index)
        
        # 3. Dynamic Role Allocation (The Swap Logic)
        # We calculate who is in a better position to attack.
        # "Better position" = Closer to the nearest food on the enemy side.
        
        food_list = self.get_food(game_state).as_list()
        
        if len(food_list) > 0:
            my_dist_to_food = min([self.get_maze_distance(my_pos, f) for f in food_list])
            teammate_dist_to_food = min([self.get_maze_distance(teammate_pos, f) for f in food_list])
        else:
            # Fallback if no food: Distance to enemy center
            enemy_center = (game_state.data.layout.width // 2, game_state.data.layout.height // 2) 
            if self.red: enemy_center = (game_state.data.layout.width - 1, game_state.data.layout.height // 2)
            
            my_dist_to_food = self.get_maze_distance(my_pos, enemy_center)
            teammate_dist_to_food = self.get_maze_distance(teammate_pos, enemy_center)

        # Logic: If I am closer to food than my partner, I am the Attacker.
        # If we are equidistant (start of game), use index as tie-breaker.
        is_attacker = False
        
        if my_dist_to_food < teammate_dist_to_food:
            is_attacker = True
        elif my_dist_to_food == teammate_dist_to_food:
            # Tie-breaker: First index attacks first
            if self.index == team_indices[0]:
                is_attacker = True

        # 4. Mode Selection
        if current_lead >= 8: 
            # Strategy: Lockdown (Winning significantly)
            self.mode = 'DEFEND'
        
        elif is_attacker:
            # Strategy: Offensive
            food_carried = game_state.get_agent_state(self.index).num_carrying
            
            # Check Retreat Conditions
            if food_carried >= 5 or (game_state.data.timeleft < 200 and food_carried > 0):
                self.mode = 'RETREAT'
            else:
                self.mode = 'ATTACK'
                
        else:
            # Strategy: Defensive (I am further away, so I defend)
            self.mode = 'DEFEND'

        # 5. Execution
        actions = game_state.get_legal_actions(self.index)
        if len(actions) > 1 and Directions.STOP in actions:
            actions.remove(Directions.STOP)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # --- COMMON FEATURES ---
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        active_defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer <= 0]
        scared_defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer > 0]

        # Danger Logic
        if len(active_defenders) > 0:
            min_dist = min([self.get_maze_distance(my_pos, d.get_position()) for d in active_defenders])
            features['distance_to_ghost'] = min_dist
            if min_dist <= 1:
                features['danger'] = 1
            elif min_dist <= 2: 
                features['danger'] = 0.5 

        # --- MODE SPECIFIC FEATURES ---
        if self.mode == 'ATTACK':
            food_list = self.get_food(successor).as_list()
            features['successor_score'] = -len(food_list) 

            if len(food_list) > 0:
                min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = min_distance

            capsules = self.get_capsules(successor)
            if len(capsules) > 0:
                min_cap_dist = min([self.get_maze_distance(my_pos, c) for c in capsules])
                features['distance_to_capsule'] = min_cap_dist

            if len(scared_defenders) > 0:
                 min_scared_dist = min([self.get_maze_distance(my_pos, d.get_position()) for d in scared_defenders])
                 if min_scared_dist < 3: 
                     features['eat_scared_ghost'] = min_scared_dist

        elif self.mode == 'RETREAT':
            dist_to_home = self.get_maze_distance(my_pos, self.start)
            features['distance_to_home'] = dist_to_home
            
            if len(active_defenders) > 0:
                min_dist = min([self.get_maze_distance(my_pos, d.get_position()) for d in active_defenders])
                features['too_close'] = max(0, 4 - min_dist)

        elif self.mode == 'DEFEND':
            features['on_defense'] = 1
            if my_state.is_pacman: features['on_defense'] = 0

            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            features['num_invaders'] = len(invaders)

            if len(invaders) > 0:
                # Chase logic
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists)
                
                # Active Interception (Capsule Protection)
                capsules_defending = self.get_capsules_you_are_defending(successor)
                if len(capsules_defending) > 0:
                    min_invader_capsule_dist = 9999
                    threatened_capsule = None
                    for capsule in capsules_defending:
                        for invader in invaders:
                            dist = self.get_maze_distance(invader.get_position(), capsule)
                            if dist < min_invader_capsule_dist:
                                min_invader_capsule_dist = dist
                                threatened_capsule = capsule
                    
                    if threatened_capsule and min_invader_capsule_dist < 5:
                        features['protect_capsule'] = self.get_maze_distance(my_pos, threatened_capsule)

            else:
                # PROACTIVE GUARDING with Coordination
                team_indices = self.get_team(successor)
                teammate_index = [i for i in team_indices if i != self.index][0]
                teammate_pos = successor.get_agent_state(teammate_index).get_position()

                # Check for Unguarded Capsules
                defended_capsules = self.get_capsules_you_are_defending(successor)
                unguarded_capsules = []
                for cap in defended_capsules:
                    if self.get_maze_distance(teammate_pos, cap) > 5:
                        unguarded_capsules.append(cap)
                
                if len(unguarded_capsules) > 0:
                    min_dist = min([self.get_maze_distance(my_pos, c) for c in unguarded_capsules])
                    features['distance_to_defended_capsule'] = min_dist
                else:
                    # Collaborative Patrol Logic
                    valid_patrols = []
                    for p in self.patrol_points:
                        if self.get_maze_distance(teammate_pos, p) > 10:
                            valid_patrols.append(p)
                    
                    if not valid_patrols:
                        valid_patrols = self.patrol_points

                    # Bias patrol towards the "center of mass" of our food
                    my_food = self.get_food_you_are_defending(successor).as_list()
                    if my_food:
                        # Calculate average Y position of our food
                        avg_y = sum([f[1] for f in my_food]) / len(my_food)
                        
                        # Find the patrol point closest to this Y-level
                        best_patrol = min(valid_patrols, key=lambda p: abs(p[1] - avg_y))
                        features['patrol_distance'] = self.get_maze_distance(my_pos, best_patrol)
                    else:
                        # Fallback if no food (rare)
                        min_patrol = min([self.get_maze_distance(my_pos, p) for p in valid_patrols])
                        features['patrol_distance'] = min_patrol
            
            if action == Directions.STOP: features['stop'] = 1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        weights = util.Counter()
        
        if self.mode == 'ATTACK':
            weights['successor_score'] = 100
            weights['distance_to_food'] = -1
            weights['distance_to_capsule'] = -50 
            weights['distance_to_ghost'] = 2 
            weights['danger'] = -1000 
            weights['eat_scared_ghost'] = -20 # FIX: Negative weight to minimize distance (Chase)

        elif self.mode == 'RETREAT':
            weights['distance_to_home'] = -2
            weights['danger'] = -1000
            weights['too_close'] = -20 

        elif self.mode == 'DEFEND':
            weights['num_invaders'] = -1000
            weights['on_defense'] = 100
            weights['invader_distance'] = -10
            weights['protect_capsule'] = -20 
            weights['distance_to_defended_capsule'] = -10
            weights['patrol_distance'] = -1
            weights['stop'] = -100
            weights['reverse'] = -2
            
        return weights

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