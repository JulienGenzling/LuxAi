from sys import stderr

from pathfinding import manhattan_distance, astar, create_weights, path_to_actions
from base import Global, is_team_sector, warp_point

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
                        # print("GATHERING ", self.match_number*101+1+self.match_step, ship.unit_id, file=stderr)