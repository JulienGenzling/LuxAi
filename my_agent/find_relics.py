from sys import stderr

from base import Global, is_team_sector
from pathfinding import (
    find_closest_target,
    astar,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    nearby_positions,
)


def find_relics(self):

    # Fix immobile ships after all nodes have been explored but no relic found in the first match.
    if (
        self.match_number == 0
        and all(node.explored_for_relic for node in self.space)
        and len(self.space.relic_nodes) == 0
    ):
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
