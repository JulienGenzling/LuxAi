from sys import stderr

from base import Global, ActionType, is_team_sector, warp_point, invert_action
from pathfinding import (
    astar,
    estimate_energy_cost,
    path_to_actions,
    create_weights,
    find_closest_target,
    manhattan_distance,
    surrounding,
)
from sap import preshot

import numpy as np


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

        if not is_team_sector(self.team_id, target_node.x, target_node.y):
            if not has_sufficient_energy(self, ship) or np.random.random() < 0.3:
                # print("NOT ENOUGH ENERGY !!!!!", file=stderr)
                return False

        ship.task = "harvest"
        ship.target = target_node
        ship.action = actions[0]
        next_node = self.space.get_node(*path[1])

        # if np.random.random() < 0.3 and not is_team_sector(self.team_id, ship.node.x, ship.node.y):
        #     for node in surrounding(self, ship.node):
        #         if node.energy == None:
        #             continue
        #         if node.coordinates == path[1]:
        #             continue
        #         if node.energy > 0 and node.type != "asteroid":
        #             ship.action = ActionType.from_coordinates(ship.coordinates, node.coordinates)

        # Verify that there is not going to be a collision with a stronger enemy ship (frequent when harvesting, around the diagonal)
        # Gagner au corps a corps
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
                # print("whiiiiiiiiiiiiiiiii", file=stderr)
                gather_energy(self, ship)
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
            # print("whoooooooooooooooooooo", file=stderr)
            gather_energy(self, ship)
            ship.task = None
            ship.target = None


def gather_energy(self, ship):
    energy_spot = find_energy_spot(self, ship)

    if energy_spot:
        path = astar(create_weights(self.space), ship.coordinates, energy_spot)
        actions = path_to_actions(path)
        if actions:
            ship.action = actions[0]
            # print(f"GATHERING: Ship {ship.unit_id} heading to energy spot at {energy_spot}", file=stderr)

    # If at energy spot, consider sapping enemy reward nodes
    if ship.energy >= Global.UNIT_SAP_COST:
        for reward_node in self.space.reward_nodes:
            # Only target enemy reward nodes
            if not is_team_sector(self.team_id, reward_node.x, reward_node.y):
                # Check if in SAP range
                in_sap_range = is_in_sap_range(
                    ship.coordinates, reward_node.coordinates
                )

                if in_sap_range:
                    # Check for enemy ships on node
                    enemies_on_node = any(
                        enemy.coordinates == reward_node.coordinates
                        for enemy in self.opp_fleet
                        if enemy.coordinates is not None
                    )

                    # SAP if there are enemies or occasionally to prevent camping
                    if enemies_on_node:
                        ship.action = ActionType.sap
                        ship.sap = reward_node.coordinates
                        # print(f"ENERGY GATHER SAP: Ship {ship.unit_id} sapping from {ship.coordinates} to {reward_node.coordinates}", file=stderr)
                        break


def has_sufficient_energy(self, ship):
    if ship.coordinates is None:
        return False

    stronger_enemy_nearby = False

    for enemy in self.opp_fleet:
        if enemy.coordinates is None:
            continue

        # Check if enemy is near target
        if enemy.energy:
            if enemy.energy > ship.energy and enemy.node.reward:
                # print("stronger enemy on reward node", file=stderr)
                stronger_enemy_nearby = True
                break

    if stronger_enemy_nearby:
        return False  # Always gather more energy if facing stronger enemies

    return True


def find_energy_spot(self, ship):
    if ship.coordinates is None:
        return None

    search_radius = 3

    best_spot = ship.node.coordinates
    best_energy = ship.node.energy

    for x_offset in range(-search_radius, search_radius + 1):
        for y_offset in range(-search_radius, search_radius + 1):

            pot_x, pot_y = warp_point(
                ship.coordinates[0] + x_offset,
                ship.coordinates[1] + y_offset,
            )
            pot_node = self.space.get_node(pot_x, pot_y)
            if pot_node.energy == None:
                continue

            if pot_node.energy > best_energy and pot_node.is_walkable:
                best_spot = pot_node.coordinates
                best_energy = pot_node.energy

    return best_spot


def is_in_sap_range(source, target):
    sap_dist = max(abs(target[0] - source[0]), abs(target[1] - source[1]))
    return sap_dist <= Global.UNIT_SAP_RANGE