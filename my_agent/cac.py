from sys import stderr

from pathfinding import manhattan_distance
from base import ActionType, invert_action, _DIRECTIONS, warp_point
from sap import preshot


def cac(self):
    for ship in self.fleet:
        action = ship.action
        if action == None:
            action = ActionType.center
        dir = _DIRECTIONS[action]
        next_node = self.space.get_node(
            *warp_point(*(ship.coordinates[0] + dir[0], ship.coordinates[1] + dir[1]))
        )
        doublecheck_action(self, ship, next_node, action)


def doublecheck_action(
    self, ship, next_node, action
):  # compute time + prendre en compte 2 mecs
    for enemy in self.opp_fleet:
        if (
            enemy.coordinates is None
            or manhattan_distance(ship.coordinates, enemy.coordinates) >= 3
        ):
            continue

        probable_coord = preshot(self, enemy)
        if (
            probable_coord == next_node.coordinates
            and enemy.energy >= ship.energy
        ):
            print(
                self.match_step + 101 * self.match_number,
                " will collide and we weak!!!",
                file=stderr,
            )
            ship.action = invert_action(action)
            return
        
        if probable_coord:
            if (
                manhattan_distance(ship.coordinates, probable_coord) == 1
                and enemy.energy < ship.energy
            ):
                print(
                    ship.unit_id,
                    self.match_step + 101 * self.match_number,
                    "Attempt to kill corps à corps",
                    file=stderr,
                )
                ship.action = ActionType.from_coordinates(ship.coordinates, probable_coord)