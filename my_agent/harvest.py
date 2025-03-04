from sys import stderr

from base import ActionType
from pathfinding import (
    astar,
    estimate_energy_cost,
    path_to_actions,
    create_weights,
    find_closest_target,
)


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
