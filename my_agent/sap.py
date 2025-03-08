from sys import stderr
import numpy as np
import random

from base import Global, ActionType, warp_point, NodeType, is_team_sector
from pathfinding import astar, find_closest_target, create_weights, nearby_positions


def sap(self, match_step, used_ship_for_dropoff):
    # Initial available ships
    available_ships = set()
    dropoff_id = used_ship_for_dropoff.unit_id if used_ship_for_dropoff != None else None
    
    for ship in self.fleet:
        if ship.energy > Global.UNIT_SAP_COST and ship.unit_id != dropoff_id:
            if (not (ship.task == "harvest" and ship.node != ship.target)) or Global.UNIT_SAP_DROPOFF_FACTOR == 1:
                available_ships.add(ship)

    # Ne pas target les ships tests qu'on target pour calculer le dropoff factor sinon ça va casser la mesure
    if hasattr(self, '_sap_tracking'):
        dropoff_targets = self._sap_tracking["calibration_targets"]
        dropoff_target_ids = [elem[0] for elem in dropoff_targets]
    else:
        dropoff_target_ids = []

    sap_1(self, available_ships, dropoff_target_ids)
    if available_ships:
        sap_2(self, available_ships, dropoff_target_ids)  # Sap other ships

    for ship in list(available_ships): 
        if ship.task == "harvest" and ship.node != ship.target:
            available_ships.remove(ship)

    if available_ships and match_step >= 30: 
        sap_3(self, available_ships)  # Blind sap on enemy reward nodes


def sap_1(self, available_ships, dropoff_target_ids):
    # Find all enemy ship positions
    enemy_positions = [
        (enemy.coordinates[0], enemy.coordinates[1])
        for enemy in self.opp_fleet
        if enemy.unit_id not in dropoff_target_ids
    ]

    # Find clusters of enemies (ships that are adjacent to each other)
    clusters = _find_enemy_clusters(enemy_positions)

    # For each cluster, calculate best sap position
    for cluster in clusters:
        if len(cluster) < 2:  # Only care about clusters of 2+ ships
            continue

        # Calculate the optimal sap position for this cluster
        best_pos, damage_score = _calculate_optimal_sap_position(self, cluster)

        if (
            best_pos and damage_score > Global.UNIT_SAP_COST
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
            ships_to_use = min(len(ships_in_range), 1 + (damage_score > 1.5 * Global.UNIT_SAP_COST))

            # Assign ships to sap this target
            for i in range(ships_to_use):
                if i >= len(ships_in_range):
                    break

                ship, _ = ships_in_range[i]
                ship.action = ActionType.sap
                ship.sap = best_pos
                available_ships.remove(ship)



def _find_enemy_clusters(enemy_positions):
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
            preshot_pos = preshot(self, enemy_ship)
            if preshot_pos:
                predicted_positions.append((preshot_pos, enemy_ship.energy))
            else:
                # If no prediction is available, use current position
                predicted_positions.append((pos, enemy_ship.energy))
        else:
            # Fallback to current position if ship not found
            predicted_positions.append((pos, 100))

    # Consider all positions in and around the predicted positions
    for pos, energy in predicted_positions:
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
        for enemy_pos, energy in predicted_positions:
            # Calculate Manhattan distance to determine splash damage
            dist = max(
                abs(pos[0] - enemy_pos[0]),
                abs(pos[1] - enemy_pos[1]),
            )

            # Direct hit
            if dist == 0:
                score += min(Global.UNIT_SAP_COST, energy)
            # Splash damage (only if within 1 tile)
            elif dist == 1:
                score += min(Global.UNIT_SAP_COST * Global.UNIT_SAP_DROPOFF_FACTOR, energy)

        if score > best_score:
            best_score = score
            best_position = pos

    return best_position, best_score


def sap_2(self, available_ships, dropoff_target_ids):

    if self.opp_fleet:
        prioritized_enemies = []
        normal_enemies = []

        for enemy_ship in self.opp_fleet:
            if enemy_ship.unit_id in dropoff_target_ids:
                continue
            if enemy_ship.node.reward or _is_near_relic(self, enemy_ship):
                prioritized_enemies.append(enemy_ship)
            else:
                normal_enemies.append(enemy_ship)

        # Process high-value targets first, then normal targets
        for i, target_list in enumerate([prioritized_enemies, normal_enemies]):
            for enemy_ship in target_list:
                # Predict where the enemy will move to
                predicted_pos = preshot(self, enemy_ship)
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
                        max_ships_to_use = 1
                    
                    # if Global.UNIT_SAP_DROPOFF_FACTOR == 1:
                    #     max_ships_to_use += 1

                    max_ships_to_use = max(min(max_ships_to_use, enemy_ship.energy//Global.UNIT_SAP_COST + 1), 0)

                    ships_to_use = min(len(ships_in_range), max_ships_to_use)

                    # Assign ships to sap this target
                    for i in range(ships_to_use):
                        if i >= len(ships_in_range):
                            break

                        ship, _ = ships_in_range[i]
                        ship.action = ActionType.sap
                        ship.sap = predicted_pos
                        available_ships.remove(ship)


def preshot(self, ship):
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
    # # Optional early-exit: 50% chance to skip sap_3 entirely
    # if np.random.rand() < 0.5:
    #     return
    
    # 1) Gather invisible enemy nodes on the enemy's side
    invisible_enemy_nodes = [
        node for node in self.space.reward_nodes
        if (not node.is_visible)
        and (not is_team_sector(self.team_id, node.x, node.y))
    ]
    if not invisible_enemy_nodes:
        return

    # 2) Build candidate sap centers around each invisible node
    potential_positions = set()
    for node in invisible_enemy_nodes:
        x, y = node.x, node.y
        potential_positions.add((x, y))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = warp_point(x + dx, y + dy)
                potential_positions.add((nx, ny))

    if not potential_positions:
        return

    # 3) Evaluate damage for each potential position
    positions_and_damage = []
    for sap_center in potential_positions:
        cx, cy = sap_center
        total_damage = 0.0
        for node in invisible_enemy_nodes:
            (ex, ey) = (node.x, node.y)
            dist = max(abs(cx - ex), abs(cy - ey))
            # direct = 1, splash = Global.UNIT_SAP_DROPOFF_FACTOR
            if dist == 0:
                dmg = 1.0
            elif dist == 1:
                dmg = Global.UNIT_SAP_DROPOFF_FACTOR
            else:
                dmg = 0.0
            total_damage += dmg

        positions_and_damage.append((sap_center, total_damage))

    # Sort by damage descending
    positions_and_damage.sort(key=lambda x: x[1], reverse=True)
    if not positions_and_damage:
        return

    # 4) We only proceed if best_damage >= threshold
    best_damage = positions_and_damage[0][1]
    threshold = 1 + Global.UNIT_SAP_DROPOFF_FACTOR
    if best_damage < threshold:
        return

    # Gather only those positions that have damage >= threshold
    viable_positions = [pd for pd in positions_and_damage if pd[1] >= threshold]
    if not viable_positions:
        return

    # 5) For each viable position, with probability sap_chance, try to assign a sap
    for (tx, ty), dmg_val in viable_positions:

        if np.random.rand() < 0.5: # min(Global.UNIT_SAP_DROPOFF_FACTOR, .75):
            # Find all ships in range
            in_range_ships = []
            for ship in list(available_ships):
                if ship.coordinates is None or ship.energy <= Global.UNIT_SAP_COST:
                    continue
                dist = max(abs(tx - ship.coordinates[0]), abs(ty - ship.coordinates[1]))
                if dist <= Global.UNIT_SAP_RANGE:
                    in_range_ships.append((ship, dist))

            # Sort ships by distance descending, then by energy ascending (or adjust as you like)
            in_range_ships.sort(key=lambda x: (-x[1], x[0].energy))
            if not in_range_ships:
                continue

            # Pick the furthest ship to sap
            best_ship, _ = in_range_ships[0]
            best_ship.action = ActionType.sap
            best_ship.sap = (tx, ty)
            available_ships.remove(best_ship)

            break
                
# def sap_3(self, available_ships):
#     reward_nodes = list(self.space.reward_nodes)

#     if not reward_nodes:
#         return

#     invisible_reward_nodes = [node for node in reward_nodes if not node.is_visible]

#     if not invisible_reward_nodes:
#         return

#     invisible_reward_nodes = list(
#         filter(
#             lambda node: not is_team_sector(self.team_id, node.x, node.y),
#             invisible_reward_nodes,
#         )
#     )

#     for ship in list(available_ships):
#         if ship.coordinates is None or ship.energy <= Global.UNIT_SAP_COST:
#             continue

#         for target_node in invisible_reward_nodes:

#             sap_dist = max(
#                 abs(target_node.x - ship.coordinates[0]),
#                 abs(target_node.y - ship.coordinates[1]),
#             )

#             if sap_dist <= Global.UNIT_SAP_RANGE and np.random.random() < 0.5:
#                 ship.action = ActionType.sap
#                 ship.sap = (target_node.x, target_node.y)
#                 available_ships.remove(ship)
#                 break


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


def sap_4(self):
    for ship_id in list(self.opp_fleet.reward_ships.keys()):
        info = self.opp_fleet.reward_ships[ship_id]
        successful_sap = False
        enemy_pos = info["node"].coordinates
        enemy_energy = info["energy"]
        ships_in_range = []
        for ship in self.fleet:
            if ship.coordinates is None:
                continue

            sap_dist = max(
                abs(enemy_pos[0] - ship.coordinates[0]),
                abs(enemy_pos[1] - ship.coordinates[1]),
            )

            if sap_dist <= Global.UNIT_SAP_RANGE:
                ships_in_range.append((ship, sap_dist))

        ships_in_range.sort(key=lambda x: (-x[1], x[0].energy))

        ships_to_use = enemy_energy // Global.UNIT_SAP_COST

        # Assign ships to sap this target
        for i in range(ships_to_use):
            if i >= len(ships_in_range):
                break

            ship, _ = ships_in_range[i]
            ship.action = ActionType.sap
            ship.sap = enemy_pos
            successful_sap = True
            # Update the energy we think the enemy has
            info["energy"] -= Global.UNIT_SAP_COST
            print("SHOOTING ", ship_id, self.match_step+self.match_number*101, file=stderr)
        if successful_sap:
            del self.opp_fleet.reward_ships[ship_id]