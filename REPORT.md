# Pacman Capture the Flag: AI Strategy Report

## 1. Executive Summary
Our solution implements a **Hybrid Reflex Agent** architecture (`SmartAgent`). To overcome the challenges of partial observability and dynamic opponent behavior, the agent combines **Probabilistic State Estimation** (Particle Filters) with **Feature-Based Decision Making**. The team coordinates using dynamic role allocation (Attacker/Defender) and employs a Finite State Machine (FSM) to switch between tactical modes such as "Berserker Attack," "Safe Retreat," and "Zone Defense."

---

## 2. State Estimation: Tracking Invisible Enemies
A critical challenge in this domain is the "Fog of War." To address this, we implemented a **Particle Filter** to infer the locations of unobserved opponents.

### 2.1 The Belief Distribution
Each agent maintains a belief distribution (`self.beliefs`) representing the probability of an enemy occupying specific coordinates on the map.
* **Initialization:** At the start of the game, probabilities are uniformly distributed across all legal tile positions.

### 2.2 The Update Cycle
The `update_beliefs` method refines these probabilities every game tick:
1.  **Exact Observation:** If an enemy is within line-of-sight (5 units), the distribution collapses to a single point (Probability = 1.0).
2.  **Transition Model (Movement):** We simulate potential enemy movement by spreading the probability mass from a tile to its neighbors, accounting for the uncertainty of the enemy's action.
3.  **Sensor Model (Noisy Distances):** The game provides a noisy Manhattan distance to opponents. We update the belief at position $P$ using Bayes' rule:
    $$P(Pos|Reading) \propto P(Reading|Pos) \times P(Pos)$$
4.  **Negative Information:** If an enemy is *not* seen within the agent's 5-tile visual radius, we zero out the probabilities for all visible tiles, significantly pruning the search space.

---

## 3. Strategic Decision Making
The agents do not rely on fixed roles. Instead, they utilize **Dynamic Role Allocation** and varying operational modes based on the game state (score, time remaining, and carrying capacity).

### 3.1 Dynamic Role Allocation
At the start of every turn, the agents communicate implicitly to decide who attacks and who defends:
* **Distance Heuristic:** The agent closest to the enemy food (or the center boundary) assumes the **Attacker** role. The teammate assumes the **Defender** role.
* **The "Catenaccio" Trigger:** If our team holds a significant lead (Score > 6), both agents switch to `DEFEND` mode to run out the clock, minimizing the risk of counter-scores.

### 3.2 Zone Defense
To prevent defensive redundancy (e.g., two ghosts chasing one invader while leaving the other side open), we implemented a **Zone Defense** system:
* **Chokepoint Analysis:** The agent calculates valid entry points along the central map boundary.
* **Map Splitting:** The map is divided vertically. Based on the agent's index, one is assigned the **Top Patrol Zone** and the other the **Bottom Patrol Zone**.

### 3.3 Finite State Machine (Modes)
Based on the allocation, the agent enters one of three modes:
1.  **ATTACK:** Aggressively invades enemy territory to collect food.
2.  **RETREAT:** Prioritizes returning home to score. Triggered when:
    * Food carried $\ge$ 5.
    * Game time < 200 ticks.
    * An active ghost is closing in (and no power capsule is available).
3.  **DEFEND:** Patrols specific chokepoints or intercepts detected invaders.

---

## 4. Tactical Execution (Features & Weights)
The agent selects actions by maximizing a linear value function: 
$$Value(action) = \sum (w_i \times f_i)$$

### 4.1 Attack Tactics
* **Food Clustering:** Instead of greedily targeting the single nearest dot, the agent calculates the average distance to the closest $k$ food pellets. This encourages the agent to target clusters, maximizing efficiency.
* **Safe Pathfinding (Safe BFS):** We implemented a custom `get_safe_bfs_distance` algorithm. Unlike standard distance metrics, this treats tiles adjacent to predicted ghost positions as "walls," allowing the agent to pathfind *around* threats rather than simply fleeing away from them.
* **"Berserker" Mode:** If a power capsule is closer than an approaching ghost, the agent ignores the `danger` penalty. This allows the agent to bait the ghost, eat the capsule, and immediately turn the tables.

### 4.2 Retreat Tactics
* **Opportunistic Snacking:** While retreating, the agent scans for food that is strictly along the path home. If picking up an extra dot does not decrease the safety margin, the agent will collect it before scoring.
* **Panic Handling:** If the path home is blocked by a ghost (distance = infinity), the `danger` weight spikes, forcing the agent to prioritize survival (jinking/reversing) over scoring.

### 4.3 Defense Tactics
* **Invader Interception:** The primary defensive goal is minimizing the Manhattan distance to the *most likely* position of the nearest invader.
* **Emergency Capsule Guarding:** If an invader is within 5 units of a power capsule, the defender applies a massive penalty to the invader-capsule distance, prioritizing positioning itself between the enemy and the power-up.

---

## 5. Conclusion
The `SmartAgent` succeeds by balancing risk and reward. The **Particle Filter** mitigates the lack of information, **Zone Defense** ensures map coverage, and the **Safe BFS** allows for intelligent navigation around threats. This combination results in an agent that is aggressive when safe, cautious when threatened, and highly coordinated with its teammate.