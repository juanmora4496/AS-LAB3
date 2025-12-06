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
        
        self.legal_positions = []
        self.beliefs = {}

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        self.legal_positions = [p for p in game_state.get_walls().as_list(False)]
        self.distancer.get_maze_distances()
        
        # Initialize the particle filter to track enemies
        self.opponents = self.get_opponents(game_state)
        for opp in self.opponents:
            self.beliefs[opp] = util.Counter()
            for p in self.legal_positions:
                self.beliefs[opp][p] = 1.0 / len(self.legal_positions)

        # Identify chokepoints to improve our patrol patterns
        self.calculate_chokepoints(game_state)

        # Establish patrol zones (Top/Bottom)
        team_indices = self.get_team(game_state)
        team_indices.sort()
        self.patrol_points.sort(key=lambda p: p[1])
        split = len(self.patrol_points) // 2
        
        if self.index == team_indices[0]:
            self.my_patrol_points = self.patrol_points[:split]
        else:
            self.my_patrol_points = self.patrol_points[split:]
            
        if not self.my_patrol_points:
            self.my_patrol_points = self.patrol_points

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
        
        # Filter for tunnels if we have too many points
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
            
            # If we can see the enemy, update belief to exact position
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos is not None:
                new_belief[opp_pos] = 1.0
                self.beliefs[opp] = new_belief
                continue

            # Transition Model: enemies move
            for p in self.legal_positions:
                if self.beliefs[opp][p] > 0:
                    x, y = p
                    possibilities = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x, y)]
                    valid_moves = [pos for pos in possibilities if pos in self.legal_positions]
                    
                    prob = self.beliefs[opp][p] / len(valid_moves)
                    for next_pos in valid_moves:
                        new_belief[next_pos] += prob
            
            # Sensor Model: noisy distance reading
            observed_dist = noisy_distances[opp]
            for p in self.legal_positions:
                true_dist = util.manhattan_distance(my_pos, p)
                
                # If we're defending, we can rule out positions on the wrong side
                if self.mode == 'DEFEND':
                    if self.red and p[0] > (game_state.data.layout.width // 2):
                        new_belief[p] = 0
                        continue
                    if not self.red and p[0] < (game_state.data.layout.width // 2):
                        new_belief[p] = 0
                        continue

                prob_obs = game_state.get_distance_prob(true_dist, observed_dist)
                new_belief[p] *= prob_obs

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
        
        # Update game state info
        current_lead = self.get_score(game_state)
        team_indices = self.get_team(game_state)
        
        my_pos = game_state.get_agent_position(self.index)
        teammate_index = [i for i in team_indices if i != self.index][0]
        teammate_pos = game_state.get_agent_position(teammate_index)
        
        # Decide who attacks and who defends
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

        # Choose a high-level strategy
        # Return flag if we can win or are carrying a lot
        food_carried = game_state.get_agent_state(self.index).num_carrying
        if (current_lead + food_carried) > 7 or food_carried > 0:
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

        # Ultra Attacking Mode, if losing and time is running out 
        if current_lead < 0 and game_state.data.timeleft < 200:
            ghost_nearby = any([self.get_maze_distance(my_pos, self.get_most_likely_position(o)) < 5 for o in self.opponents])
            
            # Retain greed unless ghost is near
            threshold = 5 if not ghost_nearby else 2 
            if food_carried >= threshold:
                self.mode = 'ULTRA_RETREAT'
            else:
                self.mode = 'ULTRA_ATTACK'

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

        # Infer enemy positions
        enemies_indices = self.get_opponents(successor)
        active_ghosts_pos = []
        invaders_pos = []
        
        for idx in enemies_indices:
            enemy_state = successor.get_agent_state(idx)
            pos = enemy_state.get_position()
            if pos is None:
                pos = self.get_most_likely_position(idx)
            
            if enemy_state.is_pacman:
                invaders_pos.append(pos)
            elif enemy_state.scared_timer <= 2: # Treat almost-recovered ghosts as dangerous
                active_ghosts_pos.append(pos)

        # --- ATTACK / ULTRA_ATTACK MODE ---
        if self.mode == 'ATTACK' or self.mode == 'ULTRA_ATTACK':
            features['successor_score'] = 0

            if self.mode == 'ULTRA_ATTACK':
                # Split the map so we don't step on each other's toes
                food_list = self.get_food(successor).as_list()
                
                # Top/Bottom split based on agent index
                team_indices = self.get_team(game_state)
                team_indices.sort()
                is_bottom = (self.index == team_indices[0])
                
                mid_y = game_state.data.layout.height // 2
                
                my_food = []
                if is_bottom:
                    my_food = [f for f in food_list if f[1] < mid_y]
                else:
                    my_food = [f for f in food_list if f[1] >= mid_y]
                    
                # If cleared our side, help the teammate
                if not my_food:
                    my_food = food_list
                    
                features['successor_score'] = -len(my_food)
                
                # Cluster food to avoid eating isolated dots first
                if len(my_food) > 0:
                    k = min(len(my_food), 3)
                    closest_k = sorted(my_food, key=lambda f: self.get_maze_distance(my_pos, f))[:k]
                    avg_dist = sum([self.get_maze_distance(my_pos, f) for f in closest_k]) / k
                    features['food_cluster_dist'] = avg_dist

                # Safe Path to My Food
                if len(my_food) > 0:
                    dist_to_food = self.get_safe_bfs_distance(successor, my_pos, my_food, active_ghosts_pos)
                    if dist_to_food < 9000:
                        features['distance_to_food'] = dist_to_food

            else: # Normal ATTACK
                food_list = self.get_food(successor).as_list()
                features['successor_score'] = -len(food_list) 
                
                # Prefer clusters of food
                if len(food_list) > 0:
                    k = min(len(food_list), 3)
                    closest_k = sorted(food_list, key=lambda f: self.get_maze_distance(my_pos, f))[:k]
                    avg_dist = sum([self.get_maze_distance(my_pos, f) for f in closest_k]) / k
                    features['food_cluster_dist'] = avg_dist 
                
                # Safe Path to Food
                if len(food_list) > 0:
                    dist_to_food = self.get_safe_bfs_distance(successor, my_pos, food_list, active_ghosts_pos)
                    if dist_to_food < 9000:
                        features['distance_to_food'] = dist_to_food
            
            # Shared Logic: Eating capsules and avoiding ghosts
            if len(self.get_capsules(successor)) < len(current_capsules):
                features['eat_capsule'] = 1
            
            # Identify threats
            obstacles = active_ghosts_pos.copy()
            capsules = self.get_capsules(successor)
            
            # Check distances
            dist_to_capsule = 9999
            if len(capsules) > 0:
                dist_to_capsule = self.get_safe_bfs_distance(successor, my_pos, capsules, obstacles)
            
            dist_to_ghost = 9999
            if len(active_ghosts_pos) > 0:
                dist_to_ghost = min([self.get_maze_distance(my_pos, p) for p in active_ghosts_pos])

            # If we can reach a capsule before a ghost reaches us, go for it
            if dist_to_capsule < dist_to_ghost:
                features['distance_to_capsule'] = dist_to_capsule
            else:
                # Otherwise, play it safe
                if dist_to_capsule < 9000: 
                    features['distance_to_capsule'] = dist_to_capsule
                
                if dist_to_ghost <= 1: 
                    features['danger'] = 1
                elif dist_to_ghost <= 2:
                    features['danger'] = 0.5

        elif self.mode == 'RETREAT' or self.mode == 'ULTRA_RETREAT':
            dist_to_home = self.get_safe_bfs_distance(successor, my_pos, [self.start], active_ghosts_pos)
            
            if dist_to_home < 9000:
                features['distance_to_home'] = dist_to_home
            else:
                features['distance_to_home'] = self.get_maze_distance(my_pos, self.start)
                features['danger'] = 1 # Path is blocked, panic!
            
            # Avoid ghosts on the way home
            if len(active_ghosts_pos) > 0:
                min_dist = min([self.get_maze_distance(my_pos, p) for p in active_ghosts_pos])
                if min_dist <= 1: features['danger'] = 1
            
            # Grab food on the way back if it's safe
            food_list = self.get_food(successor).as_list()
            if len(food_list) > 0:
                my_dist_home = self.get_maze_distance(my_pos, self.start)
                safe_food = []
                for f in food_list:
                    # Only consider food that is 'forward' towards home
                    if self.get_maze_distance(f, self.start) < my_dist_home:
                        # Check safety: No ghost within 3 steps of this food
                        is_safe = True
                        for g in active_ghosts_pos:
                            if self.get_maze_distance(f, g) <= 3:
                                is_safe = False; break
                        if is_safe: safe_food.append(f)
                
                if safe_food:
                    # Minimize distance to this safe food
                    features['distance_to_safe_food'] = min([self.get_maze_distance(my_pos, f) for f in safe_food])

        elif self.mode == 'DEFEND':
            features['on_defense'] = 1
            if my_state.is_pacman: features['on_defense'] = 0

            features['num_invaders'] = len(invaders_pos)

            if len(invaders_pos) > 0:
                dists = [self.get_maze_distance(my_pos, p) for p in invaders_pos]
                features['invader_distance'] = min(dists)
                
                # Defend the capsules
                capsules_defending = self.get_capsules_you_are_defending(successor)
                if capsules_defending:
                    min_inv_cap_dist = min([self.get_maze_distance(inv, cap) 
                                            for inv in invaders_pos 
                                            for cap in capsules_defending])
                    if min_inv_cap_dist < 5:
                        features['emergency_capsule_guard'] = 1
            else:
                # Patrol Chokepoints
                target_points = self.my_patrol_points if hasattr(self, 'my_patrol_points') else self.patrol_points
                if target_points:
                    min_patrol = min([self.get_maze_distance(my_pos, p) for p in target_points])
                    features['patrol_distance'] = min_patrol

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        weights = util.Counter()
        
        if self.mode == 'ATTACK':
            weights['successor_score'] = 200
            weights['distance_to_food'] = -2
            weights['food_cluster_dist'] = -1
            weights['distance_to_capsule'] = -20
            weights['eat_capsule'] = 5000         # Big priority
            weights['danger'] = -1000
            weights['stop'] = -100
            weights['reverse'] = -2

        elif self.mode == 'RETREAT':
            weights['distance_to_home'] = -5
            weights['distance_to_safe_food'] = -2
            weights['danger'] = -1000
            weights['stop'] = -100
            weights['reverse'] = -2

        elif self.mode == 'ULTRA_ATTACK':
            weights['successor_score'] = 500
            weights['distance_to_food'] = -5
            weights['food_cluster_dist'] = -1
            weights['distance_to_capsule'] = -20
            weights['eat_capsule'] = 5000
            weights['danger'] = -10 # Ignore fear (mostly)
            weights['stop'] = -500
            weights['reverse'] = -10

        elif self.mode == 'ULTRA_RETREAT':
            weights['distance_to_home'] = -50
            weights['distance_to_safe_food'] = 0
            weights['danger'] = -100
            weights['stop'] = -500
            weights['reverse'] = -10

        elif self.mode == 'DEFEND':
            weights['num_invaders'] = -1000
            weights['on_defense'] = 100
            weights['invader_distance'] = -10
            weights['emergency_capsule_guard'] = -200
            weights['patrol_distance'] = -1
            weights['stop'] = -100
            weights['reverse'] = -2
            
        return weights

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor