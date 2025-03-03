import numpy as np
from sys import stderr

from base import (
    Global,
    NodeType,
    ActionType,
    SPACE_SIZE,
    get_match_step,
    warp_point,
    is_team_sector,
    get_match_number,
)
from debug import show_map, show_energy_field, show_exploration_map
from pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
)
from node import Node
from space import Space
from fleet import Fleet

from find_relics import find_relics
from find_rewards import find_rewards
from harvest import harvest
from gather_energy import gather_energy
from sap import sap
from check import check
from dropoff import calibrate_sap_dropoff_factor

class Agent:

    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)
        self.match_number = None
        self.match_step = None

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.match_step = get_match_step(step)
        self.match_number = get_match_number(step)

        if self.match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)
            if self.match_number <= Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                self.space.clear_exploration_info()
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

                
        print(f"[match {self.match_number}| step {self.match_step}]", file=stderr)
        find_relics(self)
        find_rewards(self)
        harvest(self)
        gather_energy(self)
        if Global.UNIT_SENSOR_RANGE >= 3 and self.match_number <= 1:
            used_ship_for_dropoff = calibrate_sap_dropoff_factor(self)
            sap(self, self.match_step, used_ship_for_dropoff)
        else:
            sap(self, self.match_step, None)
        check(self)
        return self.create_actions_array()

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                sap_dx, sap_dy = 0, 0
                if ship.action == ActionType.sap:
                    sap_dx, sap_dy = self._get_sap_offset(ship)
                actions[i] = [ship.action, sap_dx, sap_dy]

        return actions

    def _get_sap_offset(self, ship):
        """Calculate the offset for sapping direction."""
        return ship.sap[0] - ship.coordinates[0], ship.sap[1] - ship.coordinates[1]

    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)
