from sys import stderr
from base import Global, NodeType, ActionType, SPACE_SIZE
from pathfinding import manhattan_distance
from sap import preshot


def calibrate_sap_dropoff_factor(self):

    # Initialize tracking system if it doesn't exist
    if not hasattr(self, "_sap_tracking"):
        self._sap_tracking = {
            "prev_turn_ships": {},  # Positions and energy of enemy ships from previous turn
            "calibration_targets": [],  # Enemy ships we're targeting for calibration
            "calibration_complete": False,  # Flag to indicate if we've determined the factor
        }

    # Skip if calibration is already complete
    if self._sap_tracking["calibration_complete"]:
        return

    # First, analyze any splash damage from previous turn's calibration attempts
    analyze_splash_damage(self)

    # Then, record current enemy positions for next turn's analysis
    record_enemy_positions(self)

    # Find ships for calibration targeting
    ship = find_calibration_targets(self)

    return ship


def analyze_splash_damage(self):

    prev_ships = self._sap_tracking["prev_turn_ships"]
    calibration_targets = self._sap_tracking["calibration_targets"]

    # Skip if we don't have previous data or calibration targets
    if not prev_ships or not calibration_targets:
        return

    # For each enemy ship that we targeted for calibration
    for enemy_id, target_sap_pos in calibration_targets:
        # Skip if this enemy wasn't visible last turn
        if enemy_id not in prev_ships:
            continue

        # Find the corresponding enemy ship in the current turn
        current_ship = None
        for ship in self.opp_fleet:
            if ship.unit_id == enemy_id:
                current_ship = ship
                break

        if not current_ship or not current_ship.node.is_visible:
            # print("Enemy ship not here ", file=stderr)
            continue
        
        for ship in self.fleet:
            if manhattan_distance(ship.coordinates, current_ship.coordinates) <= 1: # Il y aura un terme de perte d'energie du au contact entre les 2 ships qu'on sait pas vraiment prendre en compte
                # print("Too close to one of our ships !", file=stderr)
                return

        # Add a condition to verify that the effective current position of the targeted ship is at at a distance of 1 (diagonals or top bottom left right) of the sap position where we sapped
        dist_to_sap = max(
            abs(current_ship.coordinates[0] - target_sap_pos[0]),
            abs(current_ship.coordinates[1] - target_sap_pos[1]),
        )
        if dist_to_sap != 1:  # Must be exactly 1 tile away (diagonal or orthogonal)
            # print("Fail the splash target test because ", current_ship.coordinates,target_sap_pos,  file=stderr)
            continue

        # Get previous position and energy
        prev_pos, prev_energy = prev_ships[enemy_id]

        # Calculate expected energy loss from movement
        expected_energy_loss = 0
        if current_ship.coordinates != prev_pos:
            expected_energy_loss += Global.UNIT_MOVE_COST
        # Subtract energy gained from new position
        if current_ship.node.energy is not None:
            # print("node energy ", current_ship.node.energy, file=stderr)
            expected_energy_loss -= current_ship.node.energy
        # Add nebula reduction if applicable
        if current_ship.node.type == NodeType.nebula:
            expected_energy_loss += Global.NEBULA_ENERGY_REDUCTION

        # Calculate actual energy loss
        # print("prev energy ", prev_energy, file=stderr)
        # print("cuurent ship energy ", current_ship.energy, file=stderr)
        actual_energy_loss = prev_energy - current_ship.energy

        # Unexpected energy loss might be from SAP
        sap_damage = actual_energy_loss - expected_energy_loss
        # print("SAP DAMAGE ", sap_damage, file=stderr)

            # Calculate dropoff factor (splash damage / full damage)
        dropoff_factor = sap_damage / Global.UNIT_SAP_COST

        # print(
        #     f"Calibration: Detected splash damage: {sap_damage}, calculated dropoff: {dropoff_factor:.2f}",
        #     file=stderr,
        # )

        possible_values = [0.25, 0.5, 1]
        closest_value = min(possible_values, key=lambda x: abs(x - dropoff_factor))

        Global.UNIT_SAP_DROPOFF_FACTOR = closest_value
        self._sap_tracking["calibration_complete"] = True
        # print(f"SAP dropoff factor calibrated to {closest_value}", file=stderr)


def record_enemy_positions(self):
    # Clear previous turn's data
    self._sap_tracking["prev_turn_ships"] = {}

    # Record all visible enemy ships
    for enemy_ship in self.opp_fleet:
        if enemy_ship.node.is_visible:
            self._sap_tracking["prev_turn_ships"][enemy_ship.unit_id] = (
                enemy_ship.coordinates,
                enemy_ship.energy,
            )


def find_calibration_targets(self):

    # Clear previous calibration targets
    self._sap_tracking["calibration_targets"] = []

    # Skip if calibration is complete or if we're already targeting ships
    if (
        self._sap_tracking["calibration_complete"]
        or len(self._sap_tracking["calibration_targets"]) > 0
    ):
        return

    # Find visible enemy ships with enough energy to survive a splash attack
    potential_targets = []
    for enemy_ship in self.opp_fleet:
        if (
            enemy_ship.node.is_visible
            and enemy_ship.energy > Global.UNIT_SAP_COST
            and enemy_ship.node.type != NodeType.asteroid
        ):
            potential_targets.append(enemy_ship)

    # Skip if no suitable targets
    if not potential_targets:
        return

    # Prioritize targets (stationary ships on rewards are ideal)
    potential_targets.sort(
        key=lambda ship: (
            1 if ship.node.reward else 0,  # Prefer ships on reward nodes
            -ship.energy,  # Then prefer ships with more energy
        ),
        reverse=True,
    )

    # For each potential target, find a ship that can SAP adjacent to it
    for enemy_ship in potential_targets:
        # Generate positions adjacent to the enemy ship
        potential_coordinates = preshot(self, enemy_ship)

        if not potential_coordinates:
            continue

        potential_dx = potential_coordinates[0] - enemy_ship.coordinates[0]
        potential_dy = potential_coordinates[1] - enemy_ship.coordinates[1]
        dx = -1 if potential_dx > 0 else -1
        dy = -1 if potential_dy > 0 else -1

        adj_x = (potential_coordinates[0] + dx) % SPACE_SIZE
        adj_y = (potential_coordinates[1] + dy) % SPACE_SIZE

        adj_pos = (adj_x, adj_y)
        node = self.space.get_node(adj_x, adj_y)
        if node.type == NodeType.asteroid: # Etre sur un asteroid pourrait biaiser la mesure
            continue

        for ship in self.fleet:
            if ship.coordinates is None or ship.energy < Global.UNIT_SAP_COST:
                continue

            # Calculate distance to the adjacent position
            sap_dist = max(
                abs(adj_pos[0] - ship.coordinates[0]),
                abs(adj_pos[1] - ship.coordinates[1]),
            )

            # If in SAP range and has enough energy, assign the ship to calibrate
            if sap_dist <= Global.UNIT_SAP_RANGE:
                ship.action = ActionType.sap
                ship.sap = adj_pos

                # Record this calibration attempt
                self._sap_tracking["calibration_targets"].append(
                    (enemy_ship.unit_id, adj_pos)
                )
                # print(
                #     f"Calibration timestep {self.match_number*101+self.match_step+1}: Ship {ship.unit_id} targeting position {adj_pos} adjacent to enemy {enemy_ship.unit_id}",
                #     file=stderr,
                # )
                return ship
