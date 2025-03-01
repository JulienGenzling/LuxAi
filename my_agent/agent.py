import numpy as np
from sys import stderr

from base import (
    Global,
    NodeType,
    ActionType,
    SPACE_SIZE,
    get_match_step,
    warp_point,
    is_team_sector,
    get_match_number,
)
from debug import show_map, show_energy_field, show_exploration_map
from pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)
from node import Node
from space import Space
from fleet import Fleet

# from find_relic import find_relic
# from find_reward import find_reward
# from harvest import harvest
# from gather_energy import gather_energy
# from check import check

class Agent:

    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)
        self.match_number = None
        self.match_step = None

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.match_step = get_match_step(step)
        self.match_number = get_match_number(step)

        if self.match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)
            if self.match_number <= Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                self.space.clear_exploration_info()
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        self.find_relics()
        self.find_rewards()
        self.harvest()
        # self.gather_energy()
        # self.sap()
        # self.check()
        return self.create_actions_array()

    def gather_energy(self):
        """
        Make ships gather energy before venturing into enemy territory.
        Ships targeting enemy reward nodes will first gather energy if they
        don't have at least 100 energy and are already in enemy territory.
        The gathering position should be within sap range of enemy reward nodes.
        """
        # Minimum energy threshold before venturing deep into enemy territory
        MIN_ENERGY_THRESHOLD = 150

        for ship in self.fleet:
            # Skip ships that are not targeting harvest or have no coordinates
            if (
                ship.task != "harvest"
                or ship.coordinates is None
                or ship.target is None
                or (ship.task == "harvest" and ship.target == ship.node) # ship deja en train d'harvest
                or manhattan_distance(ship.coordinates, ship.target.coordinates) < Global.UNIT_SAP_RANGE // 1.5 # Deja trop proche du reward pour s'arrêter la
            ):
                continue

            # Check if the ship is targeting an enemy reward node
            target_in_enemy_territory = not is_team_sector(
                self.team_id, *ship.target.coordinates
            )

            # Check if the ship is already in enemy territory
            ship_in_enemy_territory = not is_team_sector(
                self.team_id, *ship.coordinates
            )

            # Only apply this strategy if ship is in enemy territory and targeting enemy reward
            if (
                ship_in_enemy_territory
                and target_in_enemy_territory
                and ship.energy < MIN_ENERGY_THRESHOLD
            ):
                # Look for high energy spots that are:
                # 1. Close to the ship's current position
                # 2. Within sap range of the target reward node
                # 3. Preferably in a safer position

                best_energy_spot = None
                best_energy_value = 0
                max_search_radius = (
                    5  # Limit search radius to avoid excessive computation
                )

                # Search for energy nodes around current position
                for x_offset in range(-max_search_radius, max_search_radius + 1):
                    for y_offset in range(-max_search_radius, max_search_radius + 1):
                        # Skip if search radius is too large (Manhattan distance)
                        if abs(x_offset) + abs(y_offset) > max_search_radius:
                            continue

                        # Calculate potential energy node position
                        pot_x, pot_y = warp_point(
                            ship.coordinates[0] + x_offset,
                            ship.coordinates[1] + y_offset,
                        )
                        pot_node = self.space.get_node(pot_x, pot_y)

                        # Skip if the node is not walkable or has no energy
                        if (
                            not pot_node.is_walkable
                            or pot_node.energy is None
                            or pot_node.energy <= 0
                        ):
                            continue

                        # Check if within sap range of the target
                        target_dist = max(
                            abs(pot_x - ship.target.coordinates[0]),
                            abs(pot_y - ship.target.coordinates[1]),
                        )

                        # Must be within sap range and have decent energy
                        if (
                            target_dist <= Global.UNIT_SAP_RANGE
                            and pot_node.energy > best_energy_value
                        ):
                            # Prefer spots in our territory if possible
                            safety_bonus = (
                                5 if is_team_sector(self.team_id, pot_x, pot_y) else 0
                            )

                            # Calculate total value considering energy and safety
                            total_value = pot_node.energy + safety_bonus

                            if total_value > best_energy_value:
                                best_energy_value = total_value
                                best_energy_spot = (pot_x, pot_y)

                # If we found a good energy spot, reroute the ship
                if best_energy_spot:
                    energy_node = self.space.get_node(*best_energy_spot)

                    # Calculate path to energy spot
                    path = astar(
                        create_weights(self.space), ship.coordinates, best_energy_spot
                    )

                    if path:
                        # Change the ship's immediate task to gather energy
                        actions = path_to_actions(path)
                        if actions:
                            ship.action = actions[0]
                            # Temporarily change task but keep the original target
                            print("GATHERING ", self.match_number*101+1+self.match_step, ship.unit_id, file=stderr)

    def check(self):
        main_targets = set()  # reward nodes
        secondary_targets = set()  # around relic nodes

        main_targets_occupied = set()
        secondary_targets_occupied = set()

        # Get the already occupied nodes
        for ship in self.fleet.ships:
            if ship.coordinates != None:
                node = self.space.get_node(*ship.coordinates)
                if node.reward:
                    main_targets_occupied.add(node)
                elif node.relic_proximity:
                    secondary_targets_occupied.add(node)

        for node in self.space:
            if node.reward and node not in main_targets_occupied:
                main_targets.add(node)
            elif node.relic_proximity and node not in secondary_targets_occupied:
                secondary_targets.add(node)

        for ship in self.fleet.ships:
            if ship.energy > Global.UNIT_MOVE_COST and ship.action == None:
                # print(match_number*100+match_step, " | ", ship.unit_id, " | ", ship.coordinates, file=stderr)
                if ship.node in main_targets and ship.node not in main_targets_occupied:
                    # print("ALREADY ON A REWARD NODE", file=stderr)
                    ship.task = "harvest"
                    ship.target = ship.node
                    continue

                target, _ = find_closest_target(
                    ship.coordinates,
                    [n.coordinates for n in main_targets],
                )
                # print("target found : ", target, file=stderr)
                if target:
                    # print("Veut aller chercher main target en ", target, file=stderr)
                    node = self.space.get_node(*target)
                    main_targets.remove(node)
                    # print("Found a target : ", target, file=stderr)
                    action = self._find_best_single_action(ship, target)
                    ship.action = action
                    # print("Resulting action : ", action, file=stderr)
                elif target is None and ship.node.coordinates:
                    # if ship.node in secondary_targets or ship.node in secondary_targets_occupied:
                    #     # print("ALREADY ON A SECONDARY NODE", file=stderr)
                    #     continue
                    target, _ = find_closest_target(
                        ship.coordinates,
                        [n.coordinates for n in secondary_targets],
                    )
                    if target:
                        # print("Veut aller chercher secondary target en ", target, file=stderr)
                        node = self.space.get_node(*target)
                        secondary_targets.remove(node)
                        action = self._find_best_single_action(ship, target)
                        # print("Resulting action ", action, file=stderr)
                        ship.action = action

    def _find_best_single_action(self, ship, target):
        x, y = ship.coordinates

        if ship.coordinates is None:
            return

        if x == target[0] and y == target[1]:
            return

        # Determine movement candidates based on ship's position relative to target
        if x > target[0] and y > target[1]:  # target is top-left
            candidates = {ActionType.left: (x - 1, y), ActionType.up: (x, y - 1)}
        elif x < target[0] and y > target[1]:  # target is top-right
            candidates = {ActionType.right: (x + 1, y), ActionType.up: (x, y - 1)}
        elif x > target[0] and y < target[1]:  # target is bottom-left
            candidates = {ActionType.left: (x - 1, y), ActionType.down: (x, y + 1)}
        elif x < target[0] and y < target[1]:  # target is bottom-right
            candidates = {ActionType.right: (x + 1, y), ActionType.down: (x, y + 1)}
        else:  # Target is directly aligned on one axis
            if x > target[0]:  # target is directly left
                return ActionType.left
            elif x < target[0]:  # target is directly right
                return ActionType.right
            elif y > target[1]:  # target is directly above
                return ActionType.up
            elif y < target[1]:  # target is directly below
                return ActionType.down

        # print(candidates, file=stderr)
        # Determine the best action among the candidates
        best_action = None
        best_energy = float("-inf")
        asteroid_count = sum(
            1
            for coord in candidates.values()
            if self.space.get_node(coord[0], coord[1]).type == "asteroid"
        )

        for action, coord in candidates.items():
            node = self.space.get_node(coord[0], coord[1])

            if node is None or node.type is None:
                continue  # Skip if node is invalid

            if asteroid_count == 2:
                return ActionType.center  # Both are asteroids

            if node.type == "asteroid":
                continue  # Skip this action if it's an asteroid

            if node.energy is not None and node.energy > best_energy:
                best_energy = node.energy
                best_action = action

        return best_action

    def sap(self):

        # Initial available ships
        available_ships = set()
        for ship in self.fleet:
            if ship.energy > Global.UNIT_SAP_COST and not (
                ship.task == "harvest" and ship.node != ship.target
            ):
                available_ships.add(ship)

        self.sap_1(available_ships)
        if available_ships:
            self.sap_2(available_ships)  # Sap other ships
        if available_ships and self.match_step >= 30:
            self.sap_3(available_ships)  # Blind sap on enemy reward nodes

    def sap_1(self, available_ships):
        # Find all enemy ship positions
        enemy_positions = [
            (enemy.coordinates[0], enemy.coordinates[1]) for enemy in self.opp_fleet
        ]

        # Find clusters of enemies (ships that are adjacent to each other)
        clusters = self._find_enemy_clusters(enemy_positions)

        # For each cluster, calculate best sap position
        for cluster in clusters:
            if len(cluster) < 2:  # Only care about clusters of 2+ ships
                continue

            # Calculate the optimal sap position for this cluster
            best_pos, damage_score = self._calculate_optimal_sap_position(cluster)

            if (
                best_pos and damage_score > 1.0
            ):  # Only worth it if we can hit more than one ship effectively
                # Find ships that can hit this position
                ships_in_range = []
                for ship in list(available_ships):
                    if ship.coordinates is None:
                        continue

                    sap_dist = max(
                        abs(best_pos[0] - ship.coordinates[0]),
                        abs(best_pos[1] - ship.coordinates[1]),
                    )

                    if sap_dist <= Global.UNIT_SAP_RANGE:
                        ships_in_range.append((ship, sap_dist))

                # Sort by distance (furthest first) to maximize range advantage
                ships_in_range.sort(key=lambda x: (-x[1], x[0].energy))

                # Use 1-2 ships depending on cluster size and damage potential
                ships_to_use = min(len(ships_in_range), 1 + (damage_score > 1.5))

                # Assign ships to sap this target
                for i in range(ships_to_use):
                    if i >= len(ships_in_range):
                        break

                    ship, _ = ships_in_range[i]
                    ship.action = ActionType.sap
                    ship.sap = best_pos
                    available_ships.remove(ship)

    def _find_enemy_clusters(self, enemy_positions):
        clusters = []
        visited = set()

        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = warp_point(x + dx, y + dy)
                neighbor_pos = (nx, ny)
                if neighbor_pos in enemy_positions:
                    neighbors.append(neighbor_pos)
            return neighbors

        def dfs(pos, current_cluster):
            visited.add(pos)
            current_cluster.append(pos)

            for neighbor in get_neighbors(pos):
                if neighbor not in visited:
                    dfs(neighbor, current_cluster)

        # Find all connected groups of enemy ships
        for pos in enemy_positions:
            if pos not in visited:
                current_cluster = []
                dfs(pos, current_cluster)
                clusters.append(current_cluster)

        return clusters

    def _calculate_optimal_sap_position(self, cluster):
        # Get all potential sap positions (all positions within UNIT_SAP_RANGE of any ship we have)
        potential_positions = set()

        # Get predicted positions for each enemy in the cluster
        predicted_positions = []
        for pos in cluster:
            # Find the enemy ship at this position
            enemy_ship = None
            for ship in self.opp_fleet:
                if ship.coordinates == pos:
                    enemy_ship = ship
                    break

            if enemy_ship:
                # Get the predicted position
                preshot_pos = self._preshot(enemy_ship)
                if preshot_pos:
                    predicted_positions.append(preshot_pos)
                else:
                    # If no prediction is available, use current position
                    predicted_positions.append(pos)
            else:
                # Fallback to current position if ship not found
                predicted_positions.append(pos)

        # Consider all positions in and around the predicted positions
        for pos in predicted_positions:
            x, y = pos
            # Center position
            potential_positions.add((x, y))

            # Adjacent positions
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = warp_point(x + dx, y + dy)
                    potential_positions.add((nx, ny))

        best_position = None
        best_score = 0

        # For each potential position, calculate the damage score using predicted positions
        for pos in potential_positions:
            score = 0
            for enemy_pos in predicted_positions:
                # Calculate Manhattan distance to determine splash damage
                dist = max(
                    abs(pos[0] - enemy_pos[0]),
                    abs(pos[1] - enemy_pos[1]),
                )

                # Direct hit
                if dist == 0:
                    score += 1.0
                # Splash damage (only if within 1 tile)
                elif dist == 1:
                    score += Global.UNIT_SAP_DROPOFF_FACTOR

            if score > best_score:
                best_score = score
                best_position = pos

        return best_position, best_score

    def sap_2(self, available_ships):

        if self.opp_fleet:
            prioritized_enemies = []
            normal_enemies = []

            for enemy_ship in self.opp_fleet:
                if enemy_ship.node.reward or self._is_near_relic(enemy_ship):
                    prioritized_enemies.append(enemy_ship)
                else:
                    normal_enemies.append(enemy_ship)

            # Process high-value targets first, then normal targets
            for i, target_list in enumerate([prioritized_enemies, normal_enemies]):
                for enemy_ship in target_list:
                    # Predict where the enemy will move to
                    predicted_pos = self._preshot(enemy_ship)
                    if predicted_pos:  # there is a preshot

                        # Sort available ships by distance to target (furthest first for range advantage)
                        ships_in_range = []
                        for ship in list(available_ships):
                            if ship.coordinates is None:
                                continue

                            sap_dist = max(
                                abs(predicted_pos[0] - ship.coordinates[0]),
                                abs(predicted_pos[1] - ship.coordinates[1]),
                            )

                            if sap_dist <= Global.UNIT_SAP_RANGE:
                                ships_in_range.append((ship, sap_dist))

                        # Sort by distance (furthest first) to maximize range advantage
                        ships_in_range.sort(key=lambda x: (-x[1], x[0].energy))

                        if (
                            i == 1 and len(ships_in_range) < 2
                        ):  # normal enemy ship and not many ships in range
                            continue
                        # Determine how many ships to use based on enemy value and our resources
                        # More aggressive approach: use more ships against high-value targets
                        if i == 0:
                            max_ships_to_use = 3 if enemy_ship.node.reward else 2
                        else:
                            max_ships_to_use = 2 if enemy_ship.node.reward else 1

                        ships_to_use = min(len(ships_in_range), max_ships_to_use)

                        # Assign ships to sap this target
                        for i in range(ships_to_use):
                            if i >= len(ships_in_range):
                                break

                            ship, _ = ships_in_range[i]
                            ship.action = ActionType.sap
                            ship.sap = predicted_pos
                            available_ships.remove(ship)

    def _preshot(self, ship):
        # Si l'enemy ship est sur un reward node, on considère qu'il est immobile
        x, y = ship.coordinates
        if ship.node.reward:
            # print("Preshot ",  (x,y), file=stderr)
            return ship.coordinates

        # Si l'ennemi est sur un asteroide, ne pas lui tirer dessus
        if ship.node.type == NodeType.asteroid:
            return None

        target, _ = find_closest_target(
            ship.coordinates,
            [n.coordinates for n in self.space.reward_nodes],
        )
        if not target:
            return None

        path = astar(create_weights(self.space), ship.coordinates, target)
        if not path:
            return None
        else:
            probable_coordinates = path[1]

        return probable_coordinates

    def sap_3(self, available_ships):
        reward_nodes = list(self.space.reward_nodes)

        if not reward_nodes:
            return

        invisible_reward_nodes = [node for node in reward_nodes if not node.is_visible]

        if not invisible_reward_nodes:
            return

        invisible_reward_nodes = list(
            filter(
                lambda node: not is_team_sector(self.team_id, node.x, node.y),
                invisible_reward_nodes,
            )
        )

        for ship in list(available_ships):
            if ship.coordinates is None or ship.energy <= Global.UNIT_SAP_COST:
                continue

            for target_node in invisible_reward_nodes:

                sap_dist = max(
                    abs(target_node.x - ship.coordinates[0]),
                    abs(target_node.y - ship.coordinates[1]),
                )

                if sap_dist <= Global.UNIT_SAP_RANGE and np.random.random() < 0.5:
                    ship.action = ActionType.sap
                    ship.sap = (target_node.x, target_node.y)
                    available_ships.remove(ship)
                    break

    def _get_sap_offset(self, ship):
        """Calculate the offset for sapping direction."""
        return ship.sap[0] - ship.coordinates[0], ship.sap[1] - ship.coordinates[1]

    def _is_near_relic(self, ship):
        """Check if a ship is near a relic (within reward range)."""
        if ship.coordinates is None:
            return False

        for relic_node in self.space.relic_nodes:
            x, y = relic_node.coordinates
            for xy in nearby_positions(x, y, Global.RELIC_REWARD_RANGE):
                if ship.coordinates == xy:
                    return True
        return False

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                sap_dx, sap_dy = 0, 0
                if ship.action == ActionType.sap:
                    sap_dx, sap_dy = self._get_sap_offset(ship)
                actions[i] = [ship.action, sap_dx, sap_dy]

        return actions

    def find_relics(self):

        # Fix immobile ships after step 50 dans le premier match
        if (
            self.match_number == 0
            and all(node.explored_for_relic for node in self.space)
            and len(self.space.relic_nodes) == 0
        ):
            print("CLEAR MAP ", file=stderr)
            self.space.clear_exploration_info()

        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_relics"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

                for x, y in path:
                    for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                        if xy in targets:
                            targets.remove(xy)

                return True
            # elif actions and ship.energy < energy:
            #     ship.action = actions[0]
            #     ship.task = None
            #     ship.target = None
            #     return False

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self):
        self.all_reward_found_count = 0
        if Global.ALL_REWARDS_FOUND:
            # self.all_reward_found_count += 1
            # if self.all_reward_found_count == 1:
            #     print(self.match_number*101+self.match_step+1, ' ALL REWARD FOUND', file=stderr)
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics()

        relic_node_to_ship = {}
        for ship in self.fleet:
            if ship.task == "find_rewards":
                if ship.target is None:
                    ship.task = None
                    continue

                if (
                    ship.target in unexplored_relics
                    and ship.energy > Global.UNIT_MOVE_COST * 5
                ):
                    relic_node_to_ship[ship.target] = ship
                else:
                    ship.task = None
                    ship.target = None

        for relic in unexplored_relics:
            if relic not in relic_node_to_ship:

                # find the closest ship to the relic node
                min_distance, closes_ship = float("inf"), None
                for ship in self.fleet:
                    if ship.task and ship.task != "find_rewards":
                        continue

                    if ship.energy < Global.UNIT_MOVE_COST * 5:
                        continue

                    distance = manhattan_distance(ship.coordinates, relic.coordinates)
                    if distance < min_distance:
                        min_distance, closes_ship = distance, ship

                if closes_ship:
                    relic_node_to_ship[relic] = closes_ship

        def set_task(ship, relic_node, can_pause):
            targets = []
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))

            target, _ = find_closest_target(ship.coordinates, targets)

            if target == ship.coordinates and not can_pause:
                target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        n.coordinates
                        for n in self.space
                        if n.explored_for_reward and n.is_walkable
                    ],
                )

            if not target:
                return False

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]
                return True
            # elif actions and ship.energy < energy:
            #     ship.action = actions[0]
            #     ship.task = None
            #     ship.target = None
            #     return False
            else:
                return False

        can_pause = True
        for n, s in sorted(
            list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id
        ):
            if set_task(s, n, can_pause):
                if s.target == s.node:
                    can_pause = False
                    pass
            else:
                if s.task == "find_rewards":
                    s.task = None
                    s.target = None

    def get_unexplored_relics(self) -> list[Node]:
        relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)

        return relic_nodes

    def harvest(self):

        def set_task(ship, target_node):
            if ship.node == target_node:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            # elif actions and ship.energy < energy:
            #     ship.action = actions[0]
            #     ship.task = None
            #     ship.target = None
            #     return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None

    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)
