from sys import stderr
from pathfinding import manhattan_distance, astar, create_weights, path_to_actions
from base import Global, is_team_sector, warp_point, ActionType, SPACE_SIZE
import numpy as np

def gather_energy(self):
    """
    Make ships gather energy before venturing into enemy territory.
    Uses logical positioning to find good energy gathering spots.
    """
    for ship in self.fleet:
        if ship.coordinates is None:
            continue
            
        # Skip ships that are not targeting harvest, have no target, or are already harvesting
        if (
            (ship.task != None and ship.task != "harvest")
            or ship.target is None
            or (ship.task == "harvest" and ship.target == ship.node)
        ):
            continue

        # Check if the ship is targeting an enemy reward node
        target_in_enemy_territory = not is_team_sector(
            self.team_id, *ship.target.coordinates
        )

        # Only consider energy gathering when heading to enemy territory
        if target_in_enemy_territory:
            # Check if ship is getting close to target
            approaching_target = manhattan_distance(ship.coordinates, ship.target.coordinates) < Global.UNIT_SAP_RANGE * 2
            
            # Check if ship has enough energy
            has_enough_energy = has_sufficient_energy(self, ship)
            
            # Gather energy if approaching target without enough energy
            if approaching_target and not has_enough_energy:
                # Find a good energy gathering spot
                energy_spot = find_energy_spot(self, ship)
                
                if energy_spot:
                    # Navigate to energy spot
                    path = astar(
                        create_weights(self.space), ship.coordinates, energy_spot
                    )

                    if path:
                        actions = path_to_actions(path)
                        if actions:
                            ship.action = actions[0]
                            print(f"GATHERING: Ship {ship.unit_id} heading to energy spot at {energy_spot}", file=stderr)
                            
                            # If at energy spot, consider sapping enemy reward nodes
                            if can_sap(ship):
                                for reward_node in self.space.reward_nodes:
                                    # Only target enemy reward nodes
                                    if not is_team_sector(self.team_id, reward_node.x, reward_node.y):
                                        # Check if in SAP range
                                        in_sap_range = is_in_sap_range(ship.coordinates, reward_node.coordinates)
                                        
                                        if in_sap_range:
                                            # Check for enemy ships on node
                                            enemies_on_node = any(
                                                enemy.coordinates == reward_node.coordinates 
                                                for enemy in self.opp_fleet 
                                                if enemy.coordinates is not None
                                            )
                                            
                                            # SAP if there are enemies or occasionally to prevent camping
                                            if enemies_on_node or np.random.random() < 0.5:
                                                ship.action = ActionType.sap
                                                ship.sap = reward_node.coordinates
                                                print(f"ENERGY GATHER SAP: Ship {ship.unit_id} sapping from {ship.coordinates} to {reward_node.coordinates}", file=stderr)
                                                break


def has_sufficient_energy(self, ship):
    """
    Determine if a ship has enough energy based on logical factors.
    
    Returns:
        bool: True if ship has sufficient energy, False otherwise
    """
    if ship.coordinates is None or ship.target is None:
        return True
    
    # For enemy territory operations, we need energy for:
    # 1. At least two SAP shots
    # 2. Movement to get back to safety
    
    # Calculate minimum energy needed for operations
    sap_operations = Global.UNIT_SAP_COST * 2  # Two SAP operations
    moves_needed = Global.UNIT_MOVE_COST * 3   # A few moves to reposition
    
    minimum_energy = sap_operations + moves_needed
    
    # Enemy presence increases our energy needs
    enemies_nearby = False
    stronger_enemy_nearby = False
    
    for enemy in self.opp_fleet:
        if enemy.coordinates is None:
            continue
            
        # Check if enemy is near target
        if manhattan_distance(enemy.coordinates, ship.target.coordinates) <= Global.UNIT_SENSOR_RANGE:
            enemies_nearby = True
            
            # Check if enemy has more energy
            if enemy.energy > ship.energy:
                stronger_enemy_nearby = True
                break
    
    # With enemies nearby, we need more energy
    if enemies_nearby:
        minimum_energy *= 2  # Double our energy needs
        
    # With stronger enemies, we need even more
    if stronger_enemy_nearby:
        return False  # Always gather more energy if facing stronger enemies
    
    # Ship has sufficient energy if it meets or exceeds the minimum needed
    return ship.energy >= minimum_energy


def find_energy_spot(self, ship):
    """
    Find a good energy gathering position using simple criteria.
    
    Returns:
        tuple: Coordinates of the best energy spot, or None if no good spot found
    """
    if ship.coordinates is None or ship.target is None:
        return None
    
    # Use ship sensor range as search radius
    search_radius = Global.UNIT_SENSOR_RANGE
    
    # Track best spot and its energy value
    best_spot = None
    best_energy = 0
    
    # Search area around current position
    for x_offset in range(-search_radius, search_radius + 1):
        for y_offset in range(-search_radius, search_radius + 1):
            # Skip if outside search radius
            if abs(x_offset) + abs(y_offset) > search_radius:
                continue
                
            # Calculate potential position
            pot_x, pot_y = warp_point(
                ship.coordinates[0] + x_offset,
                ship.coordinates[1] + y_offset,
            )
            pot_node = self.space.get_node(pot_x, pot_y)
            
            # Skip unwalkable or energy-less nodes
            if (not pot_node.is_walkable or 
                pot_node.energy is None or 
                pot_node.energy <= 0):
                continue
            
            # Skip if there's a stronger enemy here
            position_is_safe = True
            for enemy in self.opp_fleet:
                if (enemy.coordinates == (pot_x, pot_y) and 
                    enemy.energy > ship.energy):
                    position_is_safe = False
                    break
            
            if not position_is_safe:
                continue
                
            # Skip the center area of the map
            center_x, center_y = SPACE_SIZE // 2, SPACE_SIZE // 2
            is_center_area = (
                abs(pot_x - center_x) < 5 and 
                abs(pot_y - center_y) < 5
            )
            
            if is_center_area:
                continue
            
            # Evaluate this spot
            current_energy = pot_node.energy
            
            # Prioritize our territory for safety
            if is_team_sector(self.team_id, pot_x, pot_y):
                # Our territory spots are always better
                if best_spot is None or not is_team_sector(self.team_id, best_spot[0], best_spot[1]) or current_energy > best_energy:
                    best_spot = (pot_x, pot_y)
                    best_energy = current_energy
            # Consider enemy territory spots only if we don't have any in our territory yet
            elif best_spot is None or (not is_team_sector(self.team_id, best_spot[0], best_spot[1]) and current_energy > best_energy):
                best_spot = (pot_x, pot_y)
                best_energy = current_energy
    
    return best_spot


def can_sap(ship):
    """Check if ship has enough energy to perform a SAP operation"""
    return ship.energy >= Global.UNIT_SAP_COST


def is_in_sap_range(source, target):
    """Check if target is within SAP range of source"""
    sap_dist = max(
        abs(target[0] - source[0]),
        abs(target[1] - source[1])
    )
    return sap_dist <= Global.UNIT_SAP_RANGE
