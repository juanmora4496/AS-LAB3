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
    Greedy Hybrid Agent with Particle Filter Tracking and Safe-Path BFS.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.mode = 'ATTACK'
        self.patrol_points = []
        
        # Tracking Variables
        self.legal_positions = []
        self.beliefs = {} 
        self.red_indices = []
        self.blue_indices = []

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        # 1. Map Analysis
        self.legal_positions = [p for p in game_state.get_walls().as_list(False)]
        self.distancer.get_maze_distances() 
        
        # 2. Initialize Particle Filter
        self.opponents = self.get_opponents(game_state)
        for opp in self.opponents:
            self.beliefs[opp] = util.Counter()
            for p in self.legal_positions:
                self.beliefs[opp][p] = 1.0 / len(self.legal_positions)

        # 3. Calculate Chokepoints 
        self.calculate_chokepoints(game_state)

    def calculate_chokepoints(self, game_state):
        map_width = game_state.data.layout.width
        map_height = game_state.data.layout.height
        boundary_x = (map_width // 2) - 1 if self.red else (map_width // 2)
        
        self.patrol_points = []
        for y in range(1, map_height - 1):
            if not game_state.has_wall(boundary_x, y):
                self.patrol_points.append((boundary_x, y))
        
        # Prioritize tunnels
        if len(self.patrol_points) > 4:
            tunnels = []
            for x, y in self.patrol_points:
                neighbors = 0
                if game_state.has_wall(x, y+1): neighbors += 1
                if game_state.has_wall(x, y-1): neighbors += 1
                if neighbors > 0: tunnels.append((x,y))
            if tunnels: self.patrol_points = tunnels

    def update_beliefs(self, game_state):
        """ Standard Particle Filter update """
        my_pos = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()

        for opp in self.opponents:
            new_belief = util.Counter()
            
            # 1. Observability
            opp_pos = game_state.get_agent_position(opp)
            if opp_pos is not None:
                new_belief[opp_pos] = 1.0
                self.beliefs[opp] = new_belief
                continue

            # 2. Transition (Movement)
            for p in self.legal_positions:
                if self.beliefs[opp][p] > 0:
                    x, y = p
                    possibilities = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x, y)]
                    valid_moves = [pos for pos in possibilities if pos in self.legal_positions]
                    prob = self.beliefs[opp][p] / len(valid_moves)
                    for next_pos in valid_moves:
                        new_belief[next_pos] += prob
            
            # 3. Emission (Noisy Distance)
            observed_dist = noisy_distances[opp]
            for p in self.legal_positions:
                true_dist = util.manhattan_distance(my_pos, p)
                
                # Defending Logic Pruning
                if self.mode == 'DEFEND':
                    if self.red and p[0] > (game_state.data.layout.width // 2):
                        new_belief[p] = 0; continue
                    if not self.red and p[0] < (game_state.data.layout.width // 2):
                        new_belief[p] = 0; continue

                prob_obs = game_state.get_distance_prob(true_dist, observed_dist)
                new_belief[p] *= prob_obs
                
                if util.manhattan_distance(my_pos, p) <= 5:
                    new_belief[p] = 0 # If we don't see them <= 5, they aren't there

            new_belief.normalize()
            self.beliefs[opp] = new_belief

    def get_most_likely_position(self, agent_index):
        return self.beliefs[agent_index].arg_max()

    def get_safe_bfs_distance(self, game_state, start_pos, targets, obstacles):
        """
        BFS that treats obstacles (ghosts) as walls. 
        Returns distance to nearest target in 'targets'.
        """
        if not targets: return 9999
        
        queue = util.Queue()
        queue.push((start_pos, 0))
        visited = set([start_pos])
        target_set = set(targets)
        obstacle_set = set(obstacles)
        
        # Sanity check: if we are ON an obstacle, we are already dead/in danger
        # but for BFS we just continue
        
        while not queue.is_empty():
            pos, dist = queue.pop()
            
            if pos in target_set:
                return dist
            
            x, y = int(pos[0]), int(pos[1])
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            
            for n in neighbors:
                if n not in visited and not game_state.has_wall(n[0], n[1]) and n not in obstacle_set:
                    visited.add(n)
                    queue.push((n, dist + 1))
                    
        return 9999 # Unreachable due to obstacles

    def choose_action(self, game_state):
        self.update_beliefs(game_state)
        
        # 1. Analyze Team State
        my_pos = game_state.get_agent_position(self.index)
        team_indices = self.get_team(game_state)
        teammate_index = [i for i in team_indices if i != self.index][0]
        teammate_pos = game_state.get_agent_position(teammate_index)
        
        food_list = self.get_food(game_state).as_list()
        
        # 2. Assign Attacker/Defender Roles
        # Simple logic: closest to food attacks.
        if len(food_list) > 0:
            my_dist = min([self.get_maze_distance(my_pos, f) for f in food_list])
            tm_dist = min([self.get_maze_distance(teammate_pos, f) for f in food_list])
        else:
            my_dist = 0
            tm_dist = 0
            
        is_attacker = (my_dist < tm_dist) or (my_dist == tm_dist and self.index == team_indices[0])

        # 3. Mode Switching
        scared_ghosts = any(game_state.get_agent_state(i).scared_timer > 5 for i in self.opponents)
        food_carried = game_state.get_agent_state(self.index).num_carrying
        
        if is_attacker:
            # RETREAT LOGIC
            # If we have a lot of food OR time is running out
            # OR (we have some food AND a ghost is very close/blocking path)
            if food_carried >= 8: # Greedy: Carry more!
                self.mode = 'RETREAT'
            elif game_state.data.timeleft < 200 and food_carried > 0:
                self.mode = 'RETREAT'
            else:
                self.mode = 'ATTACK'
        else:
            self.mode = 'DEFEND'
            # Intercept: If we are defending but an invader is eating our food, we might want to attack them.
            # But 'DEFEND' mode handles that.

        # 4. Action Selection
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

        # Inferred Enemy Positions
        enemies_indices = self.get_opponents(successor)
        active_ghosts_pos = []
        invaders_pos = []
        
        for idx in enemies_indices:
            enemy_state = successor.get_agent_state(idx)
            pos = enemy_state.get_position()
            if pos is None: pos = self.get_most_likely_position(idx)
            
            if enemy_state.is_pacman:
                invaders_pos.append(pos)
            elif enemy_state.scared_timer <= 2: # Treat almost-recovered ghosts as dangerous
                active_ghosts_pos.append(pos)

        # --- ATTACK MODE ---
        if self.mode == 'ATTACK':
            food_list = self.get_food(successor).as_list()
            features['successor_score'] = -len(food_list) # Eat food!

            # SAFE PATHING: Treat active ghosts as walls
            # We add neighbors of ghosts to obstacles to be extra safe against movement
            obstacles = []
            for g in active_ghosts_pos:
                obstacles.append(g)
                # Note: Adding neighbors here makes it very cautious. 
                # For "Greedy", we might just add the ghost position itself.
                # If we get stuck, we might oscillate. Let's rely on 'distance_to_ghost' for the buffer.
            
            # Distance to Food (Safe Path)
            if len(food_list) > 0:
                # Use BFS to find actual walkable distance to nearest food
                dist = self.get_safe_bfs_distance(successor, my_pos, food_list, obstacles)
                if dist < 9000:
                    features['distance_to_food'] = dist
                else:
                    # Food is blocked!
                    features['distance_to_food'] = 9999

            # Distance to Capsules (High Priority)
            capsules = self.get_capsules(successor)
            if len(capsules) > 0:
                # Same BFS check for capsules
                dist = self.get_safe_bfs_distance(successor, my_pos, capsules, obstacles)
                if dist < 9000:
                    features['distance_to_capsule'] = dist
                    # If we are very close to a capsule, huge incentive
                    if dist <= 1: 
                        features['eat_capsule'] = 1

            # Standard Ghost Repulsion (for immediate reflexes)
            if len(active_ghosts_pos) > 0:
                min_ghost_dist = min([self.get_maze_distance(my_pos, p) for p in active_ghosts_pos])
                features['distance_to_ghost'] = min_ghost_dist
                if min_ghost_dist <= 1:
                    features['danger'] = 1

        # --- RETREAT MODE ---
        elif self.mode == 'RETREAT':
            # Avoid ghosts while going home
            obstacles = active_ghosts_pos
            dist_to_home = self.get_safe_bfs_distance(successor, my_pos, [self.start], obstacles)
            
            if dist_to_home < 9000:
                features['distance_to_home'] = dist_to_home
            else:
                # Path blocked, use maze distance but panic
                features['distance_to_home'] = self.get_maze_distance(my_pos, self.start)
                features['panic'] = 1
            
            if len(active_ghosts_pos) > 0:
                min_dist = min([self.get_maze_distance(my_pos, p) for p in active_ghosts_pos])
                if min_dist <= 2: features['danger'] = 1

        # --- DEFEND MODE ---
        elif self.mode == 'DEFEND':
            features['on_defense'] = 1
            if my_state.is_pacman: features['on_defense'] = 0
            features['num_invaders'] = len(invaders_pos)

            if len(invaders_pos) > 0:
                dists = [self.get_maze_distance(my_pos, p) for p in invaders_pos]
                features['invader_distance'] = min(dists)
            else:
                if self.patrol_points:
                    min_patrol = min([self.get_maze_distance(my_pos, p) for p in self.patrol_points])
                    features['patrol_distance'] = min_patrol

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        weights = util.Counter()
        
        if self.mode == 'ATTACK':
            weights['successor_score'] = 200     # GREEDY: Eat food now
            weights['distance_to_food'] = -2     # GREEDY: Go to food fast
            weights['distance_to_capsule'] = -20 # Priority: Capsules are good
            weights['eat_capsule'] = 500         # PRIORITY: EAT IT!
            weights['distance_to_ghost'] = 2     # Keep away slightly
            weights['danger'] = -1000            # Avoid death
            weights['stop'] = -100
            weights['reverse'] = -2

        elif self.mode == 'RETREAT':
            weights['distance_to_home'] = -2
            weights['danger'] = -1000
            weights['panic'] = -1000 # If blocked, don't go that way
            weights['stop'] = -100
            weights['reverse'] = -2

        elif self.mode == 'DEFEND':
            weights['num_invaders'] = -1000
            weights['on_defense'] = 100
            weights['invader_distance'] = -10
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