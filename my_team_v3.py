# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley.

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
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class SmartAgent(CaptureAgent):
    """
    Improved Hybrid Agent with Particle Filter, Safe BFS, and Berserker Mode.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.mode = 'ATTACK'
        self.patrol_points = []
        
        # Tracking Variables
        self.legal_positions = []
        self.beliefs = {}  # Dictionary to store beliefs for each opponent

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        # 1. Map Analysis
        self.legal_positions = [p for p in game_state.get_walls().as_list(False)]
        self.distancer.get_maze_distances() # Pre-compute distances
        
        # 2. Initialize Particle Filter for Opponents
        self.opponents = self.get_opponents(game_state)
        for opp in self.opponents:
            self.beliefs[opp] = util.Counter()
            for p in self.legal_positions:
                self.beliefs[opp][p] = 1.0 / len(self.legal_positions)

        # 3. Calculate Chokepoints (Better Patrol)
        self.calculate_chokepoints(game_state)

        # 4. Zone Assignment (Zone Defense)
        team_indices = self.get_team(game_state)
        team_indices.sort()
        self.patrol_points.sort(key=lambda p: p[1]) # Sort by Y
        split = len(self.patrol_points) // 2
        
        if self.index == team_indices[0]: # First agent takes Bottom
            self.my_patrol_points = self.patrol_points[:split]
        else: # Second agent takes Top
            self.my_patrol_points = self.patrol_points[split:]
            
        if not self.my_patrol_points: # Fallback
            self.my_patrol_points = self.patrol_points
        
        # 5. Store teammate index for faster access
        team_indices = self.get_team(game_state)
        self.teammate_index = [i for i in team_indices if i != self.index][0]

    def calculate_chokepoints(self, game_state):
        """
        Identify entry points (chokepoints) along the boundary.
        """
        map_width = game_state.data.layout.width
        map_height = game_state.data.layout.height
        
        # Determine x-coordinate of the boundary
        boundary_x = (map_width // 2) - 1 if self.red else (map_width // 2)
        
        self.patrol_points = []
        for y in range(1, map_height - 1):
            if not game_state.has_wall(boundary_x, y):
                self.patrol_points.append((boundary_x, y))
        
        # Refinement: Prioritize tunnels
        if len(self.patrol_points) > 4:
            tunnels = []
            for x, y in self.patrol_points:
                neighbors = 0
                if game_state.has_wall(x, y+1): neighbors += 1
                if game_state.has_wall(x, y-1): neighbors += 1
                if neighbors > 0:
                    tunnels.append((x,y))
            if tunnels:
                self.patrol_points = tunnels

    def update_beliefs(self, game_state):
        """
        Updates the probability distribution of enemy positions.
        """
        my_pos = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()

        for opp in self.opponents:
            new_belief = util.Counter()
            
            # CASE 1: Exact Observation
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos is not None:
                new_belief[opp_pos] = 1.0
                self.beliefs[opp] = new_belief
                continue

            # CASE 2: Movement (Transition Model)
            for p in self.legal_positions:
                if self.beliefs[opp][p] > 0:
                    x, y = p
                    possibilities = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x, y)]
                    valid_moves = [pos for pos in possibilities if pos in self.legal_positions]
                    
                    prob = self.beliefs[opp][p] / len(valid_moves)
                    for next_pos in valid_moves:
                        new_belief[next_pos] += prob
            
            # CASE 3: Noisy Distance (Sensor Model)
            observed_dist = noisy_distances[opp]
            for p in self.legal_positions:
                true_dist = util.manhattan_distance(my_pos, p)
                
                # Defend Optimization: Prune impossible sides
                if self.mode == 'DEFEND':
                    if self.red and p[0] > (game_state.data.layout.width // 2):
                        new_belief[p] = 0
                        continue
                    if not self.red and p[0] < (game_state.data.layout.width // 2):
                        new_belief[p] = 0
                        continue

                prob_obs = game_state.get_distance_prob(true_dist, observed_dist)
                new_belief[p] *= prob_obs

                # Zero out visible tiles if not seen
                if util.manhattan_distance(my_pos, p) <= 5:
                    new_belief[p] = 0

            new_belief.normalize()
            self.beliefs[opp] = new_belief

    def get_most_likely_position(self, agent_index):
        return self.beliefs[agent_index].arg_max()

    def get_safe_bfs_distance(self, game_state, start_pos, targets, obstacles):
        """
        BFS that calculates distance to nearest target while avoiding obstacles.
        """
        if not targets: return 9999
        queue = util.Queue()
        queue.push((start_pos, 0))
        visited = set([start_pos])
        target_set = set(targets)
        obstacle_set = set(obstacles)
        
        while not queue.is_empty():
            pos, dist = queue.pop()
            if pos in target_set: return dist
            
            x, y = int(pos[0]), int(pos[1])
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for n in neighbors:
                if n not in visited and not game_state.has_wall(n[0], n[1]) and n not in obstacle_set:
                    visited.add(n)
                    queue.push((n, dist + 1))
        return 9999

    def choose_action(self, game_state):
        self.update_beliefs(game_state)
        
        # 1. Update Global State Info
        current_lead = self.get_score(game_state)
        team_indices = self.get_team(game_state)
        
        my_pos = game_state.get_agent_position(self.index)
        teammate_index = [i for i in team_indices if i != self.index][0]
        teammate_pos = game_state.get_agent_position(teammate_index)
        
        # 2. Dynamic Role Allocation
        food_list = self.get_food(game_state).as_list()
        
        if len(food_list) > 0:
            my_dist_to_food = min([self.get_maze_distance(my_pos, f) for f in food_list])
            teammate_dist_to_food = min([self.get_maze_distance(teammate_pos, f) for f in food_list])
        else:
            enemy_center = (game_state.data.layout.width // 2, game_state.data.layout.height // 2) 
            if self.red: enemy_center = (game_state.data.layout.width - 1, game_state.data.layout.height // 2)
            my_dist_to_food = self.get_maze_distance(my_pos, enemy_center)
            teammate_dist_to_food = self.get_maze_distance(teammate_pos, enemy_center)

        is_attacker = False
        if my_dist_to_food < teammate_dist_to_food:
            is_attacker = True
        elif my_dist_to_food == teammate_dist_to_food and self.index == team_indices[0]:
            is_attacker = True

        # 3. Mode Selection
        # Smart Endgame: If we can secure a win by returning, do it.
        food_carried = game_state.get_agent_state(self.index).num_carrying
        if (current_lead + food_carried) > 4 and food_carried > 0:
             self.mode = 'RETREAT'
        elif current_lead >= 6: 
            self.mode = 'DEFEND'
        elif is_attacker:
            ghost_nearby = any([self.get_maze_distance(my_pos, self.get_most_likely_position(o)) < 5 for o in self.opponents])
            
            # Retain greed unless ghost is near
            threshold = 5 if not ghost_nearby else 2 

            if food_carried >= threshold or (game_state.data.timeleft < 200 and food_carried > 0):
                self.mode = 'RETREAT'
            else:
                self.mode = 'ATTACK'
        else:
            self.mode = 'DEFEND'

        # 4. Execution
        actions = game_state.get_legal_actions(self.index)
        if len(actions) > 1 and Directions.STOP in actions:
            actions.remove(Directions.STOP)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        current_capsules = self.get_capsules(game_state)

        # --- ENEMY DETECTION ---
        enemies_indices = self.get_opponents(successor)
        active_ghosts_pos = []
        invaders_pos = []
        
        for idx in enemies_indices:
            enemy_state = successor.get_agent_state(idx)
            pos = enemy_state.get_position()
            if pos is None: pos = self.get_most_likely_position(idx)
            
            if enemy_state.is_pacman: invaders_pos.append(pos)
            elif enemy_state.scared_timer <= 2: active_ghosts_pos.append(pos)

        # --- ATTACK MODE ---
        if self.mode == 'ATTACK':
            food_list = self.get_food(successor).as_list()
            features['successor_score'] = -len(food_list) 

            # IMPROVEMENT: Anti-Clumping
            # If I am too close to my teammate, it's bad (unless defending)
            teammate_pos = successor.get_agent_state(self.teammate_index).get_position()
            if self.get_maze_distance(my_pos, teammate_pos) <= 2:
                features['too_crowded'] = 1
            
            # (Keep your existing Berserker/Safe BFS logic here)
            obstacles = active_ghosts_pos.copy()
            capsules = self.get_capsules(successor)
            
            dist_to_capsule = 9999
            if len(capsules) > 0:
                dist_to_capsule = self.get_safe_bfs_distance(successor, my_pos, capsules, obstacles)
            
            dist_to_ghost = 9999
            if len(active_ghosts_pos) > 0:
                dist_to_ghost = min([self.get_maze_distance(my_pos, p) for p in active_ghosts_pos])

            if dist_to_capsule < dist_to_ghost:
                features['distance_to_capsule'] = dist_to_capsule
            else:
                if dist_to_capsule < 9000: features['distance_to_capsule'] = dist_to_capsule
                if dist_to_ghost <= 1: features['danger'] = 1
                elif dist_to_ghost <= 2: features['danger'] = 0.5

            if len(food_list) > 0:
                dist_to_food = self.get_safe_bfs_distance(successor, my_pos, food_list, obstacles)
                if dist_to_food < 9000: features['distance_to_food'] = dist_to_food

        # --- RETREAT MODE ---
        elif self.mode == 'RETREAT':
            # (Keep your existing retreat logic)
            dist_to_home = self.get_safe_bfs_distance(successor, my_pos, [self.start], active_ghosts_pos)
            if dist_to_home < 9000: features['distance_to_home'] = dist_to_home
            else: 
                features['distance_to_home'] = self.get_maze_distance(my_pos, self.start)
                features['danger'] = 1
            
            if len(active_ghosts_pos) > 0:
                if min([self.get_maze_distance(my_pos, p) for p in active_ghosts_pos]) <= 1:
                    features['danger'] = 1

        # --- DEFEND MODE ---
        elif self.mode == 'DEFEND':
            features['on_defense'] = 1
            if my_state.is_pacman: features['on_defense'] = 0
            features['num_invaders'] = len(invaders_pos)

            if len(invaders_pos) > 0:
                dists = [self.get_maze_distance(my_pos, p) for p in invaders_pos]
                features['invader_distance'] = min(dists)

                # IMPROVEMENT: Smart Intercept
                # Find the patrol point closest to the *invader* (their likely exit)
                # We want to be closer to that point than we are now
                closest_invader = min(invaders_pos, key=lambda p: self.get_maze_distance(my_pos, p))
                predict_exit = min(self.patrol_points, key=lambda p: self.get_maze_distance(closest_invader, p))
                features['intercept_distance'] = self.get_maze_distance(my_pos, predict_exit)

            else:
                # Patrol logic
                target_points = self.my_patrol_points if hasattr(self, 'my_patrol_points') else self.patrol_points
                if target_points:
                    features['patrol_distance'] = min([self.get_maze_distance(my_pos, p) for p in target_points])

        # IMPROVEMENT: Jitter Reduction (Consistency)
        # Small bonus for keeping the same direction
        current_dir = game_state.get_agent_state(self.index).configuration.direction
        if action == current_dir:
            features['consistency'] = 1

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features
    
    def get_weights(self, game_state, action):
        weights = util.Counter()
        
        if self.mode == 'ATTACK':
            weights['successor_score'] = 200
            weights['distance_to_food'] = -2
            weights['distance_to_capsule'] = -20
            weights['eat_capsule'] = 5000
            weights['danger'] = -1000
            weights['too_crowded'] = -50     # <--- NEW: Penalize clumping
            weights['stop'] = -100
            weights['reverse'] = -2
            weights['consistency'] = 1       # <--- NEW: Slight bias for smooth movement

        elif self.mode == 'RETREAT':
            weights['distance_to_home'] = -5
            weights['danger'] = -1000
            weights['stop'] = -100
            weights['reverse'] = -2
            weights['consistency'] = 1

        elif self.mode == 'DEFEND':
            weights['num_invaders'] = -1000
            weights['on_defense'] = 100
            weights['invader_distance'] = -10
            weights['intercept_distance'] = -15 # <--- NEW: Priority on cutting them off
            weights['emergency_capsule_guard'] = -200
            weights['patrol_distance'] = -1
            weights['stop'] = -100
            weights['reverse'] = -2
            weights['consistency'] = 1
            
        return weights

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor