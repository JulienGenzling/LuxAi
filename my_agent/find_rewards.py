from sys import stderr

from base import Global, is_team_sector
from node import Node
from pathfinding import (
    manhattan_distance,
    nearby_positions,
    find_closest_target,
    astar,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
)


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

    unexplored_relics = get_unexplored_relics(self)

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
    for n, s in sorted(list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id):
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
