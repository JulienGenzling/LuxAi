from sys import stderr

from base import ActionType, Global
from pathfinding import find_closest_target

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
        if ship.energy > Global.UNIT_MOVE_COST and ship.action == None and ship.coordinates != None:
            # print(ship.unit_id, " needs check", file=stderr)
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
                action = _find_best_single_action(self, ship, target)
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
                    action = _find_best_single_action(self, ship, target)
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