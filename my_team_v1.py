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
        # Feature 1: Opponent tracking with noisy distances
        self.opponent_estimated_positions = {}
        # Feature 3: Movement pattern analysis
        self.opponent_history = {}

    def register_initial_state(self, game_state):
        """
        Calculates the home boundary and patrol points for defense.
        """
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        # Initialize opponent tracking
        opponents = self.get_opponents(game_state)
        for opp in opponents:
            self.opponent_history[opp] = []
        
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
        # Feature 1: Update opponent tracking with noisy distances
        self.update_opponent_tracking(game_state)
        
        # Feature 2: Detect food theft
        self.detect_stolen_food(game_state)
        
        # Feature 3: Update movement history
        self.update_opponent_history(game_state)
        
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
        if current_lead >= 7: 
            # Strategy: Lockdown (Winning significantly)
            self.mode = 'DEFEND'
        
        elif is_attacker:
            # Strategy: Offensive
            food_carried = game_state.get_agent_state(self.index).num_carrying
            
            # Check Retreat Conditions
            if food_carried >= 3 or (game_state.data.timeleft < 200 and food_carried > 0):
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
            
            # FEATURE 1: Use opponent threat level to avoid invisible ghosts
            threat = self.get_opponent_threat_level(game_state, my_pos)
            if threat > 0:
                features['invisible_threat'] = threat
            
            # NEW: Opportunistic invader chasing when on home side  
            if not my_state.is_pacman:  # We're a ghost on our home side
                invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
                if len(invaders) > 0:
                    dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                    features['chase_invader_attack_mode'] = min(dists)
            
            # NEW: Avoid blocked entry points when trying to enter enemy territory
            # If we're still on our side but about to cross, check if defenders are blocking
            if not my_state.is_pacman and len(active_defenders) > 0:
                # Calculate boundary position
                map_width = game_state.data.layout.width
                if self.red:
                    boundary_x = (map_width // 2) - 1
                else:
                    boundary_x = (map_width // 2)
                
                # Check if we're near the boundary (within 3 units)
                dist_to_boundary = abs(my_pos[0] - boundary_x)
                if dist_to_boundary <= 3:
                    # Check if any defender is near our crossing point
                    for defender in active_defenders:
                        def_pos = defender.get_position()
                        # If defender is close to the boundary near our Y position
                        if abs(def_pos[0] - boundary_x) <= 2 and abs(def_pos[1] - my_pos[1]) <= 3:
                            features['blocked_entry'] = 1
                            break
            # Prevent oscillation - penalize reversing direction
            if action == Directions.STOP: 
                features['stop'] = 1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev: 
                features['reverse'] = 1

        elif self.mode == 'RETREAT':
            dist_to_home = self.get_maze_distance(my_pos, self.start)
            features['distance_to_home'] = dist_to_home
            
            if len(active_defenders) > 0:
                min_dist = min([self.get_maze_distance(my_pos, d.get_position()) for d in active_defenders])
                features['too_close'] = max(0, 4 - min_dist)
            
            # NEW: Opportunistically collect food on the way back if it's safe
            food_list = self.get_food(successor).as_list()
            if len(food_list) > 0:
                # Find food that is on the way back (closer to home than we are)
                my_dist_to_home = self.get_maze_distance(my_pos, self.start)
                safe_food_on_path = []
                
                for food in food_list:
                    food_dist_to_home = self.get_maze_distance(food, self.start)
                    # Food must be closer to home than current position
                    if food_dist_to_home < my_dist_to_home:
                        # Check if food is safe (no defenders within 4 units)
                        is_safe = True
                        if len(active_defenders) > 0:
                            min_defender_dist = min([self.get_maze_distance(food, d.get_position()) for d in active_defenders])
                            if min_defender_dist < 4:
                                is_safe = False
                        
                        if is_safe:
                            safe_food_on_path.append(food)
                
                if len(safe_food_on_path) > 0:
                    distances = [self.get_maze_distance(my_pos, food) for food in safe_food_on_path]
                    features['safe_food_on_retreat'] = min(distances)
            
            # Prevent oscillation
            if action == Directions.STOP: 
                features['stop'] = 1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev: 
                features['reverse'] = 1

        elif self.mode == 'DEFEND':
            features['on_defense'] = 1
            if my_state.is_pacman: features['on_defense'] = 0

            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            features['num_invaders'] = len(invaders)
            
            # FEATURE 2: Priority response to food theft
            if hasattr(self, 'stolen_food_location') and self.stolen_food_location is not None:
                features['respond_to_theft'] = self.get_maze_distance(my_pos, self.stolen_food_location)

            if len(invaders) > 0:
                # Chase logic
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists)
                
                # FEATURE 3: Use predicted positions for better interception
                for invader in invaders:
                    invader_idx = [i for i in self.get_opponents(successor) 
                                   if successor.get_agent_position(i) == invader.get_position()][0]
                    predicted_pos = self.predict_opponent_position(invader_idx, game_state)
                    if predicted_pos is not None:
                        # Validate predicted position before using it
                        try:
                            predicted_dist = self.get_maze_distance(my_pos, predicted_pos)
                            features['predicted_intercept'] = predicted_dist
                            break  # Only use first prediction
                        except:
                            # Invalid position, skip this prediction
                            pass
                
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
            weights['eat_scared_ghost'] = -5 # FIX: Negative weight to minimize distance (Chase)
            weights['invisible_threat'] = -10  # FEATURE 1: Penalty for invisible threats
            weights['chase_invader_attack_mode'] = -15  # Chase invaders when on home side
            weights['blocked_entry'] = -100  # Avoid defended entry points
            weights['stop'] = -50
            weights['reverse'] = -2

        elif self.mode == 'RETREAT':
            weights['distance_to_home'] = 3
            weights['danger'] = -1000
            weights['too_close'] = -20 
            weights['stop'] = -50
            weights['reverse'] = -2
            weights['safe_food_on_retreat'] = 1 # Lower priority than getting home

        elif self.mode == 'DEFEND':
            weights['num_invaders'] = -1000
            weights['on_defense'] = 100
            weights['invader_distance'] = -10
            weights['respond_to_theft'] = -50  # FEATURE 2: High priority for theft response
            weights['predicted_intercept'] = -8  # FEATURE 3: Intercept predicted positions
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
    
    # ==================== FEATURE 1: Opponent Tracking ====================
    def update_opponent_tracking(self, game_state):
        """
        Uses noisy distances to estimate positions of invisible opponents.
        """
        my_pos = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()
        opponents = self.get_opponents(game_state)
        
        for opp_idx in opponents:
            opp_pos = game_state.get_agent_position(opp_idx)
            
            if opp_pos is not None:
                # Opponent is visible, use exact position
                self.opponent_estimated_positions[opp_idx] = opp_pos
            else:
                # Opponent is invisible, estimate using noisy distance
                noisy_dist = noisy_distances[opp_idx]
                # Store the noisy distance for now (more sophisticated estimation possible)
                self.opponent_estimated_positions[opp_idx] = ('noisy', my_pos, noisy_dist)
    
    def get_opponent_threat_level(self, game_state, position):
        """
        Calculate threat level at a position based on estimated opponent locations.
        Returns a penalty value (higher = more dangerous).
        """
        threat = 0
        opponents = self.get_opponents(game_state)
        
        for opp_idx in opponents:
            if opp_idx in self.opponent_estimated_positions:
                est_pos = self.opponent_estimated_positions[opp_idx]
                
                if isinstance(est_pos, tuple) and est_pos[0] == 'noisy':
                    # Noisy distance estimate
                    _, ref_pos, noisy_dist = est_pos
                    # If position is roughly within the noisy distance range, increase threat
                    actual_dist = self.get_maze_distance(position, ref_pos)
                    if abs(actual_dist - noisy_dist) < 6:  # Within noise range
                        threat += max(0, 10 - noisy_dist)  # Closer = more threatening
                else:
                    # Exact position known
                    opp_state = game_state.get_agent_state(opp_idx)
                    if not opp_state.is_pacman and opp_state.scared_timer <= 0:
                        # It's an active ghost
                        dist = self.get_maze_distance(position, est_pos)
                        if dist <= 5:
                            threat += (6 - dist) * 5  # Much higher penalty for close ghosts
        
        return threat
    
    # ==================== FEATURE 2: Food Theft Detection ====================
    def detect_stolen_food(self, game_state):
        """
        Detects if food has been stolen and stores the location.
        """
        self.stolen_food_location = None
        
        prev_obs = self.get_previous_observation()
        if prev_obs is None:
            return
        
        prev_food = self.get_food_you_are_defending(prev_obs).as_list()
        curr_food = self.get_food_you_are_defending(game_state).as_list()
        
        if len(prev_food) > len(curr_food):
            # Food was stolen!
            stolen = set(prev_food) - set(curr_food)
            if stolen:
                self.stolen_food_location = list(stolen)[0]  # Get first stolen food
    
    # ==================== FEATURE 3: Movement Pattern Analysis ====================
    def update_opponent_history(self, game_state):
        """
        Tracks opponent positions over time for pattern analysis.
        """
        opponents = self.get_opponents(game_state)
        
        for opp_idx in opponents:
            opp_pos = game_state.get_agent_position(opp_idx)
            if opp_pos is not None:
                # Add position to history
                if opp_idx not in self.opponent_history:
                    self.opponent_history[opp_idx] = []
                
                self.opponent_history[opp_idx].append(opp_pos)
                
                # Keep only last 5 positions to save memory
                if len(self.opponent_history[opp_idx]) > 5:
                    self.opponent_history[opp_idx].pop(0)
    
    def predict_opponent_position(self, opp_idx, game_state):
        """
        Predicts next position based on movement history.
        Returns None if insufficient data or invalid prediction.
        """
        if opp_idx not in self.opponent_history:
            return None
        
        history = self.opponent_history[opp_idx]
        if len(history) < 2:
            return None
        
        # Simple linear prediction based on last two positions
        last_pos = history[-1]
        prev_pos = history[-2]
        
        # Calculate movement vector
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        
        # Predict next position (ensure integers)
        predicted_x = int(last_pos[0] + dx)
        predicted_y = int(last_pos[1] + dy)
        predicted = (predicted_x, predicted_y)
        
        # Validate predicted position
        walls = game_state.get_walls()
        
        # Check if within grid bounds
        if predicted_x < 0 or predicted_x >= walls.width:
            return None
        if predicted_y < 0 or predicted_y >= walls.height:
            return None
        
        # Check if it's a wall
        if walls[predicted_x][predicted_y]:
            return None
        
        return predicted