from sys import stderr

from pathfinding import manhattan_distance, astar, create_weights, path_to_actions
from base import Global, is_team_sector, warp_point, ActionType
import numpy as np

def gather_energy(self):
    """
    Make ships gather energy before venturing into enemy territory or when near enemy reward nodes.
    Uses sophisticated positioning to find optimal energy gathering spots that provide strategic advantages.
    Uses dynamic energy thresholds based on game state and enemy analysis.
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

        # Calculate dynamic energy threshold based on game state and enemy analysis
        energy_threshold = calculate_energy_threshold(self, ship)

        # Check if the ship is getting close to its target
        ship_close_to_target = (
            manhattan_distance(ship.coordinates, ship.target.coordinates)
            < Global.UNIT_SAP_RANGE * 1.2
        )

        # Only apply energy gathering strategy if approaching enemy territory with insufficient energy
        if (
            target_in_enemy_territory
            and ship_close_to_target
            and ship.energy < energy_threshold
        ):
            # Find the best energy gathering spot using our strategic function
            best_energy_spot = find_best_energy_spot(self, ship)
            
            if best_energy_spot:
                # Calculate path to energy spot
                path = astar(
                    create_weights(self.space), ship.coordinates, best_energy_spot
                )

                if path:
                    actions = path_to_actions(path)
                    if actions:
                        ship.action = actions[0]
                        print(f"GATHERING: Ship {ship.unit_id} heading to energy spot at {best_energy_spot}", file=stderr)
                        
                        # Special case: If already at the energy spot, consider sapping enemy reward nodes
                        if ship.coordinates == best_energy_spot and ship.energy >= Global.UNIT_SAP_COST:
                            for reward_node in self.space.reward_nodes:
                                if not is_team_sector(self.team_id, reward_node.x, reward_node.y):
                                    sap_dist = max(
                                        abs(reward_node.coordinates[0] - ship.coordinates[0]),
                                        abs(reward_node.coordinates[1] - ship.coordinates[1])
                                    )
                                    
                                    # If in SAP range of enemy reward node, consider shooting
                                    if sap_dist <= Global.UNIT_SAP_RANGE:
                                        # Check if any enemy ships are on this reward node
                                        enemies_on_node = False
                                        for enemy in self.opp_fleet:
                                            if enemy.coordinates == reward_node.coordinates:
                                                enemies_on_node = True
                                                break
                                                
                                        # SAP if enemies present or 50% chance randomly
                                        if enemies_on_node or np.random.random() < 0.5:
                                            ship.action = ActionType.sap
                                            ship.sap = reward_node.coordinates
                                            print(f"ENERGY GATHER SAP: Ship {ship.unit_id} sapping from position {ship.coordinates} to {reward_node.coordinates}", file=stderr)
                                            break


def calculate_energy_threshold(self, ship):
    """
    Calculate a dynamic energy threshold based on game state and strategic analysis.
    
    Factors considered:
    1. Match progression (higher threshold in later stages)
    2. Enemy presence near the target
    3. Enemy energy levels
    4. Distance to target (farther targets need more energy)
    5. Nebula presence between ship and target
    
    Args:
        ship: The ship for which to calculate threshold
        
    Returns:
        int: The dynamic energy threshold value
    """
    if ship.coordinates is None or ship.target is None:
        return 100  # Default threshold
    
    # Base threshold starts at 100
    base_threshold = 100
    
    # --- Factor 1: Match progression ---
    # As the match progresses, we become more cautious
    match_progression_factor = min(1.5, 1.0 + (self.match_step / Global.MAX_STEPS_IN_MATCH) * 0.5)
    
    # --- Factor 2: Enemy presence analysis ---
    enemy_presence_factor = 1.0
    enemies_near_target = 0
    highest_enemy_energy = 0
    
    # Check a wider area around the target for enemy ships
    target_area_radius = Global.UNIT_SENSOR_RANGE + 1
    
    for enemy_ship in self.opp_fleet:
        if enemy_ship.coordinates is None:
            continue
            
        # Distance to target
        dist_to_target = manhattan_distance(
            enemy_ship.coordinates, 
            ship.target.coordinates
        )
        
        # Count enemies near target
        if dist_to_target <= target_area_radius:
            enemies_near_target += 1
            highest_enemy_energy = max(highest_enemy_energy, enemy_ship.energy)
    
    # Adjust factor based on enemy presence
    if enemies_near_target > 0:
        # More cautious with multiple enemies
        enemy_presence_factor = 1.0 + (enemies_near_target * 0.2)
        
        # Even more cautious if enemies have high energy
        if highest_enemy_energy > ship.energy:
            energy_ratio = min(2.0, highest_enemy_energy / max(1, ship.energy))
            enemy_presence_factor *= energy_ratio
    
    # --- Factor 3: Path analysis ---
    path_factor = 1.0
    
    # Calculate approximate path to target
    dist_to_target = manhattan_distance(ship.coordinates, ship.target.coordinates)
    
    # More energy needed for longer journeys
    if dist_to_target > 5:
        path_factor += min(0.5, dist_to_target * 0.05)
    
    # --- Factor 4: Nebula presence along path ---
    nebula_factor = 1.0
    
    # Estimate nebula presence along path (simplified check)
    nebula_count = 0
    check_points = []
    
    # Check points along a simplified path
    steps = max(1, dist_to_target // 2)  # Check every other point for efficiency
    for step in range(1, steps + 1):
        # Simple linear interpolation between ship and target
        progress = step / (steps + 1)
        check_x = int(ship.coordinates[0] + (ship.target.coordinates[0] - ship.coordinates[0]) * progress)
        check_y = int(ship.coordinates[1] + (ship.target.coordinates[1] - ship.coordinates[1]) * progress)
        
        # Check for nebula at this point
        check_node = self.space.get_node(check_x, check_y)
        if check_node.type == 1:  # NodeType.nebula
            nebula_count += 1
    
    # Adjust factor based on nebula presence
    if nebula_count > 0:
        nebula_factor = 1.0 + (nebula_count * 0.15)  # 15% increase per nebula encountered
    
    # --- Factor 5: SAP strategy considerations ---
    sap_strategy_factor = 1.0
    
    # If we want to maintain SAP capability, ensure enough energy for multiple SAPs
    target_in_enemy_territory = not is_team_sector(self.team_id, *ship.target.coordinates)
    if target_in_enemy_territory:
        # Reserve energy for at least 2 SAP shots plus movement
        sap_reserve = Global.UNIT_SAP_COST * 2 + Global.UNIT_MOVE_COST * 3
        sap_strategy_factor = max(1.0, sap_reserve / base_threshold)
    
    # --- Calculate final threshold ---
    threshold = base_threshold * match_progression_factor * enemy_presence_factor * path_factor * nebula_factor * sap_strategy_factor
    
    # Round to nearest 10 for cleaner values
    threshold = round(threshold / 10) * 10
    
    # Ensure threshold is within reasonable bounds
    threshold = min(300, max(80, threshold))
    
    # Log the calculation if it's notably different from base
    if abs(threshold - base_threshold) > 30:
        factors = {
            "match": round(match_progression_factor, 2),
            "enemy": round(enemy_presence_factor, 2),
            "path": round(path_factor, 2),
            "nebula": round(nebula_factor, 2),
            "sap": round(sap_strategy_factor, 2)
        }
        # print(f"Ship {ship.unit_id} energy threshold: {threshold} (factors: {factors})", file=stderr)
    
    return threshold

def find_best_energy_spot(self, ship):
    """
    Find the optimal energy gathering position for a ship based on strategic considerations:
    
    1. High energy nodes within SAP range of enemy reward nodes but outside enemy sensor range
    2. Consider confrontation opportunities against weaker enemy ships
    3. Avoid confrontation with stronger enemy ships
    
    Args:
        ship: The ship looking for energy
        
    Returns:
        tuple: Coordinates of the best energy spot, or None if no good spot found
    """
    if ship.coordinates is None or ship.target is None:
        return None
        
    # Constants for strategy
    MAX_SEARCH_RADIUS = 7  # Extended search radius
    ENEMY_ENERGY_ADVANTAGE_THRESHOLD = 0  # How much more energy an enemy needs to have for us to avoid confrontation
    
    # Store current target for reference
    target_coords = ship.target.coordinates
    
    # Find enemy reward nodes for SAP targeting
    enemy_reward_nodes = []
    for node in self.space.reward_nodes:
        if not is_team_sector(self.team_id, node.x, node.y):
            enemy_reward_nodes.append(node)
    
    # Find enemy ships for confrontation evaluation
    enemy_ships = list(self.opp_fleet)
    
    # Sort potential spots by priority scores
    potential_spots = []
    
    # Search for energy nodes around current position
    for x_offset in range(-MAX_SEARCH_RADIUS, MAX_SEARCH_RADIUS + 1):
        for y_offset in range(-MAX_SEARCH_RADIUS, MAX_SEARCH_RADIUS + 1):
            # Skip if search radius is too large (Manhattan distance)
            if abs(x_offset) + abs(y_offset) > MAX_SEARCH_RADIUS:
                continue
                
            # Calculate potential energy node position
            pot_x, pot_y = warp_point(
                ship.coordinates[0] + x_offset,
                ship.coordinates[1] + y_offset,
            )
            pot_node = self.space.get_node(pot_x, pot_y)
            
            # Skip if the node is not walkable or has no energy
            if (not pot_node.is_walkable or 
                pot_node.energy is None or 
                pot_node.energy <= 0):
                continue
            
            # Calculate base score from energy value
            energy_score = pot_node.energy
            
            # Factor 1: Safety score - prefer positions in our territory
            safety_score = 8 if is_team_sector(self.team_id, pot_x, pot_y) else -50
            
            # Factor 2: SAP range score - check if we're within SAP range of any enemy reward nodes
            sap_score = 0
            for reward_node in enemy_reward_nodes:
                dist_to_reward = max(
                    abs(pot_x - reward_node.x),
                    abs(pot_y - reward_node.y)
                )
                
                if dist_to_reward <= Global.UNIT_SAP_RANGE:
                    sap_score += 10  # Strong bonus for being in SAP range
                
            # Factor 3: Sensor avoidance score - prefer positions outside enemy sensor range
            stealth_score = 0
            for enemy_ship in enemy_ships:
                if enemy_ship.coordinates is None:
                    continue
                    
                dist_to_enemy = max(
                    abs(pot_x - enemy_ship.coordinates[0]),
                    abs(pot_y - enemy_ship.coordinates[1])
                )
                
                # If we're outside enemy sensor range but in SAP range of reward
                if dist_to_enemy > Global.UNIT_SENSOR_RANGE and sap_score > 0:
                    stealth_score += 15  # Significant bonus for stealth SAP positions
            
            # Factor 4: Confrontation evaluation
            confrontation_score = 0
            for enemy_ship in enemy_ships:
                if enemy_ship.coordinates is None:
                    continue
                    
                if (pot_x, pot_y) == enemy_ship.coordinates:
                    # Direct confrontation - compare energy levels
                    energy_advantage = ship.energy - enemy_ship.energy
                    
                    if energy_advantage >= 0:
                        # We have equal or more energy - good for confrontation
                        confrontation_score += 20 + energy_advantage  # Strong bonus for advantageous confrontation
                    elif energy_advantage > -ENEMY_ENERGY_ADVANTAGE_THRESHOLD:
                        # Enemy has a bit more energy but not too much - neutral
                        confrontation_score += 0
                    else:
                        # Enemy has much more energy - avoid confrontation
                        confrontation_score -= 50  # Strong penalty for disadvantageous confrontation
            
            # Calculate total score
            total_score = (
                energy_score +
                safety_score +
                sap_score +
                stealth_score +
                confrontation_score
            )
            
            # Add to potential spots
            potential_spots.append((pot_x, pot_y, total_score))
    
    # Sort by score (highest first)
    potential_spots.sort(key=lambda x: x[2], reverse=True)
    
    # Return best spot or None if no spots found
    if potential_spots:
        best_x, best_y, score = potential_spots[0]
        best_node = self.space.get_node(best_x, best_y)
        if score > 0:
            print(f"Energy gather: Ship {ship.unit_id} going to ({best_x}, {best_y}) with score {score}", file=stderr)
            return (best_x, best_y)
            
    return None