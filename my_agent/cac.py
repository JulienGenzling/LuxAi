from sys import stderr

from pathfinding import manhattan_distance
from base import ActionType, invert_action, _DIRECTIONS, warp_point
from sap import preshot
from check import _find_best_single_action


def cac(self):
    for ship in self.fleet:
        if not ship.node.reward:
            action = ship.action
            if action == None:
                action = ActionType.center
            dir = _DIRECTIONS[action]
            next_node = self.space.get_node(
                *warp_point(*(ship.coordinates[0] + dir[0], ship.coordinates[1] + dir[1]))
            )
            doublecheck_action(self, ship, next_node, action)


def doublecheck_action(self, ship, next_node, action):
    coords_dict = {}  # This is initialized as an empty dict with no default values
    for enemy in self.opp_fleet:
        if (
            enemy.coordinates is None
            or manhattan_distance(ship.coordinates, enemy.coordinates) >= 3
        ):
            continue

        probable_coord = None
        if probable_coord is None:
            target = (0, 0) if self.team_id == 0 else (23, 23)
            best_single_action = _find_best_single_action(self, enemy, target)
            if best_single_action is None:
                continue
            dir = _DIRECTIONS[best_single_action]
            probable_coord = (
                enemy.coordinates[0] + dir[0],
                enemy.coordinates[1] + dir[1],
            )

        # Initialize or update the dictionary entry
        if probable_coord in coords_dict:
            coords_dict[probable_coord] += enemy.energy
        else:
            coords_dict[probable_coord] = enemy.energy

    for probable_coord, energy in coords_dict.items():
        if probable_coord == next_node.coordinates and energy >= ship.energy:
            ship.action = invert_action(
                action
            )  # régler cas immobile on reward node : bouger et sap
            return

        if (
            manhattan_distance(ship.coordinates, probable_coord) == 1
            and energy < ship.energy
        ):
            ship.action = ActionType.from_coordinates(ship.coordinates, probable_coord)
