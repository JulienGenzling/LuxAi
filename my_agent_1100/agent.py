import copy
import numpy as np
from sys import stderr
from scipy.signal import convolve2d

from base import (
    Global,
    NodeType,
    ActionType,
    SPACE_SIZE,
    get_match_step,
    warp_point,
    get_opposite,
    is_team_sector,
    get_match_number,
)
from debug import show_map, show_energy_field, show_exploration_map
from pathfinding import (
    astar,
    find_closest_target,
    find_closest_ship,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False

        self._relic = False
        self._reward = False
        self._relic_proximity = False
        self._explored_for_relic = False
        self._explored_for_reward = True

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        return self._relic
    
    @property
    def relic_proximity(self):
        return self._relic_proximity

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: None | bool):
        if self._explored_for_relic and self._relic and not status:
            raise ValueError(
                f"Can't change the relic status {self._relic}->{status} for {self}"
                ", the tile has already been explored"
            )

        if status is None:
            self._explored_for_relic = False
            return

        self._relic = status
        self._explored_for_relic = True
    
    def update_relic_proximity_status(self, status: None | bool):
        self._relic_proximity = status

    def update_reward_status(self, status: None | bool):
        if self._explored_for_reward and self._reward and not status:
            raise ValueError(
                f"Can't change the reward status {self._reward}->{status} for {self}"
                ", the tile has already been explored"
            )

        if status is None:
            self._explored_for_reward = False
            return

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Space:
    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        # set of nodes with a relic
        self._relic_nodes: set[Node] = set()

        # set of nodes that provide points
        self._reward_nodes: set[Node] = set()

        # set of nodes around a relic that don't provide points
        self._relic_proximity: set[Node] = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes
    
    def relic_proximity(self) -> set[Node]:
        return self._relic_proximity

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def update(self, step, obs, team_id, team_reward):
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(step, obs, team_id, team_reward)

    def _update_relic_map(self, step, obs, team_id, team_reward):
        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                # We have found a new relic.
                self._update_relic_status(*xy, status=True)
                for x, y in nearby_positions(*xy, Global.RELIC_REWARD_RANGE):
                    if not self.get_node(x, y).reward:
                        self._update_reward_status(x, y, status=None)

        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)

            if not node.explored_for_relic:
                all_relics_found = False

            if not node.explored_for_reward:
                all_rewards_found = False

        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found

        match = get_match_number(step)
        match_step = get_match_step(step)
        num_relics_th = 2 * min(match, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1

        if not Global.ALL_RELICS_FOUND:
            if len(self._relic_nodes) >= num_relics_th:
                # all relics found, mark all nodes as explored for relics
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)

        if not Global.ALL_REWARDS_FOUND:
            if (
                match_step > Global.LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR
                or len(self._relic_nodes) >= num_relics_th
            ):
                self._update_reward_status_from_relics_distribution()
                self._update_reward_results(obs, team_id, team_reward)
                self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        # We will use Global.REWARD_RESULTS to identify which nodes yield points
        for result in Global.REWARD_RESULTS:

            unknown_nodes = set()
            known_reward = 0
            for n in result["nodes"]:
                if n.explored_for_reward and not n.reward:
                    continue

                if n.reward:
                    known_reward += 1
                    continue

                unknown_nodes.add(n)

            if not unknown_nodes:
                # all nodes already explored, nothing to do here
                continue

            reward = result["reward"] - known_reward  # reward from unknown_nodes

            if reward == 0:
                # all nodes are empty
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=False)

            elif reward == len(unknown_nodes):
                # all nodes yield points
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=True)

            elif reward > len(unknown_nodes):
                # We shouldn't be here, but sometimes we are. It's not good.
                # Maybe I'll fix it later.
                pass

    def _update_reward_results(self, obs, team_id, team_reward):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                ship_nodes.add(self.get_node(*position))

        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1

        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1

        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)

        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)
            for x, y in nearby_positions(
                *node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                close_node = self.get_node(x, y)
                if close_node:
                    self._relic_proximity.add(close_node)
                    close_node.update_relic_proximity_status(status)
            for x, y in nearby_positions(
                *opp_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                close_node = self.get_node(x, y)
                if close_node:
                    self._relic_proximity.add(close_node)
                    close_node.update_relic_proximity_status(status)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # rewards are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)

        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)
            if node in self._relic_proximity:
                self._relic_proximity.remove(node)
            if opp_node in self._relic_proximity:
                self._relic_proximity.remove(opp_node)


    def _update_map(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]

        obstacles_shifted = False
        energy_nodes_shifted = False
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]

            if (
                is_visible
                and not node.is_unknown
                and node.type.value != obs_tile_type[x, y]
            ):
                obstacles_shifted = True

            if (
                is_visible
                and node.energy is not None
                and node.energy != obs_energy[x, y]
            ):
                energy_nodes_shifted = True

        Global.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)

        def clear_map_info():
            for n in self:
                n.type = NodeType.unknown

        if not Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Global.OBSTACLE_MOVEMENT_DIRECTION = direction

                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                clear_map_info()

        if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = self._find_obstacle_movement_period(
                Global.OBSTACLES_MOVEMENT_STATUS
            )
            if period is not None:
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Global.OBSTACLE_MOVEMENT_PERIOD = period

            if obstacles_shifted:
                clear_map_info()

        if (
            obstacles_shifted
            and Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
        ):
            # maybe something is wrong
            clear_map_info()

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            node.is_visible = is_visible

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_node(*get_opposite(x, y)).type = node.type

            if is_visible:
                node.energy = int(obs_energy[x, y])

                # the energy field should be symmetrical
                self.get_node(*get_opposite(x, y)).energy = node.energy

            elif energy_nodes_shifted:
                # The energy field has changed
                # I cannot predict what the new energy field will be like.
                node.energy = None

    @staticmethod
    def _find_obstacle_movement_period(obstacles_movement_status):
        if len(obstacles_movement_status) < 81:
            return

        num_movements = sum(obstacles_movement_status)

        if num_movements <= 2:
            return 40
        elif num_movements <= 4:
            return 20
        elif num_movements <= 8:
            return 10
        else:
            return 20 / 3

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)

            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (
                    sensor_mask[x, y]
                    and not node.is_unknown
                    and obs_tile_type[x, y] != node.type.value
                ):
                    match = False
                    break

            if match:
                suitable_directions.append(direction)

        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def clear(self):
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step):
        if (
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and Global.OBSTACLE_MOVEMENT_PERIOD > 0
        ):
            speed = 1 / Global.OBSTACLE_MOVEMENT_PERIOD
            if (step - 2) * speed % 1 > (step - 1) * speed % 1:
                self.move(*Global.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self

    def clear_exploration_info(self):
        Global.REWARD_RESULTS = []
        Global.ALL_RELICS_FOUND = False
        Global.ALL_REWARDS_FOUND = False
        for node in self:
            if not node.relic:
                self._update_relic_status(node.x, node.y, status=None)


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None
        self.sap: Node | None = None

    def __repr__(self):
        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None
        self.sap = None


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0  # how many points have we scored in this match so far
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space):
        self.points = int(obs["team_points"][self.team_id])

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()


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

        # for ship in self.fleet:
        #     if ship.task == None:
                # if self.match_number == 0:
                #     print("DE BASE NO TASK : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
        self.find_relics()
        self.find_rewards()
        self.harvest()
        self.sap(enemy_coordinates=obs["units"]["position"][1 - self.team_id])
        self.check()
        
        return self.create_actions_array()

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
        
        # print(match_number*100 + match_step, main_targets_occupied, file=stderr)

        for node in self.space:
            if node.reward and node not in main_targets_occupied:
                main_targets.add(node)
            elif node.relic_proximity and node not in secondary_targets_occupied:
                secondary_targets.add(node)

        for ship in self.fleet.ships:
            if ship.energy > Global.UNIT_MOVE_COST and ship.action == None:
                # print(match_number*100+match_step, " | ", ship.unit_id, " | ", ship.coordinates, file=stderr)
                if ship.node in main_targets or ship.node in main_targets_occupied:
                    # print("ALREADY ON A REWARD NODE", file=stderr)
                    ship.task = "harvest"
                    ship.target = ship.node
                    continue
                # print(
                #     "USELESS SHIP",
                #     " | match number : ",
                #     match_number,
                #     " | match step : ",
                #     match_step,
                #     " | unit id : ",
                #     ship.unit_id,
                #     " | task : ",
                #     ship.task,
                #     " | target : ",
                #     ship.target,
                #     " | coordinates : ",
                #     ship.coordinates,
                #     " | energy : ",
                #     ship.energy,
                #     file=stderr,
                # )
                # S'ils sont useless, ils doivent se rapprocher des rewards
                # Trouver la case reward la plus proche
                # Try to find a main target (reward node)
                target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        n.coordinates
                        for n in main_targets
                    ],
                )
                # print("target found : ", target, file=stderr)
                if target:
                    # print("Veut aller chercher main target en ", target, file=stderr)
                    node = self.space.get_node(*target)
                    main_targets.remove(node)
                    # print("Found a target : ", target, file=stderr)
                    action = self._find_best_single_action(ship, target)
                    ship.action = action
                    # print("Resulting action : ", action, file=stderr)
                elif target is None and ship.node.coordinates:
                    # if ship.node in secondary_targets or ship.node in secondary_targets_occupied:
                    #     # print("ALREADY ON A SECONDARY NODE", file=stderr)
                    #     continue
                    target, _ = find_closest_target(
                            ship.coordinates,
                        [
                            n.coordinates
                            for n in secondary_targets
                        ],
                    )
                    if target:
                        # print("Veut aller chercher secondary target en ", target, file=stderr)
                        node = self.space.get_node(*target)
                        secondary_targets.remove(node)
                        action = self._find_best_single_action(ship, target)
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

    def sap(self, enemy_coordinates):

        # sap enemy if we are not moving (action = center)
        for enemy_ship in self.opp_fleet:
            preshot_xy = self._preshot(enemy_ship)
            for ship in self.fleet.ships:
                sap_dist = np.inf
                if ship.coordinates != None:
                    sap_dist = max(
                        abs(preshot_xy[0] - ship.coordinates[0]),
                        abs(preshot_xy[1] - ship.coordinates[1]),
                    )
                # We retrieve the position of the closest ship and tell him to sap
                if (
                    sap_dist < Global.UNIT_SAP_RANGE
                    and ship.energy > Global.UNIT_SAP_COST
                    and ship.action == ActionType.center
                ):
                    # print("Ship : ", ship.unit_id, "Distance : ", dist, "Sap range : ", Global.UNIT_SAP_RANGE, "Sap where : ", xy, file=stderr)
                    ship.action = ActionType.sap
                    ship.sap = preshot_xy
    
    def _is_near_relic(self, ship):
        for relic_node in self.space.relic_nodes:
            x, y = relic_node.coordinates
            for xy in nearby_positions(x, y, Global.RELIC_REWARD_RANGE):
                if ship.coordinates == xy:
                    return True 
        return False

    def _preshot(self, ship):
        # Si l'enemy ship est sur un reward node, on considère qu'il est immobile
        x, y = ship.coordinates
        if ship.node.reward:
            # print("Preshot ",  (x,y), file=stderr)
            return ship.coordinates

        target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        n.coordinates
                        for n in self.space.reward_nodes
                    ],
                )
        if not target:
            return (x,y)
        
        # probable_action = self._find_best_single_action(ship, target)

        path = astar(create_weights(self.space), ship.coordinates, target)
        energy = estimate_energy_cost(self.space, path)
        actions = path_to_actions(path)
        if actions and actions[0] == ActionType.center: # Louche qu'on lui dise de rester au centre
            probable_action = self._find_best_single_action(self, ship, target)
        elif actions and actions[0] != ActionType.center:
            probable_action = actions[0]
        else:
            return (x,y)

        # With the action and the current position, find the resulting position
        if probable_action == ActionType.up:
            probable_coordinates = (x,y-1)
        elif probable_action == ActionType.down:
            probable_coordinates = (x,y+1)
        elif probable_action == ActionType.left:
            probable_coordinates = (x-1,y)
        elif probable_action == ActionType.right:
            probable_coordinates = (x+1,y)
        else:
            probable_coordinates = (x,y)
        # print("Preshot ", probable_coordinates, file=stderr)

        return probable_coordinates
    
    def _get_sap_offset(self, ship):
        return ship.sap[0] - ship.coordinates[0], ship.sap[1] - ship.coordinates[1]

    def _is_getting_reward(self, ship):
        for node in self.space.reward_nodes:
            if ship.coordinates != None:
                if ship.coordinates[0] == node.x and ship.coordinates[1] == node.y:
                    return True
        return False

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                sap_dx, sap_dy = 0, 0
                if ship.action == ActionType.sap:
                    sap_dx, sap_dy = self._get_sap_offset(ship)
                actions[i] = [ship.action, sap_dx, sap_dy]
            # else:
                # if self.match_number == 0:
                #     print("NO TASK FOUND : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)

        return actions

    def find_relics(self):
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                # We will only find relics in our part of the map
                # because relics are symmetrical.
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                # if self.match_number == 0:
                #     print("OTHER TASK RELIC : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                # if self.match_number == 0:
                #     print("NO ENERGY RELIC : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                # if self.match_number == 0:
                #     print("NO TARGET FOUND RELIC : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
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
            # else:
                # if self.match_number == 0:
                #     print("NOT ACTION IN RELIC : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                # if self.match_number == 0:
                #     print("PIPI : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                ship.task = None
                ship.target = None

    def find_rewards(self):
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics()

        relic_node_to_ship = {}
        for ship in self.fleet:
            if ship.task == "find_rewards":
                if ship.target is None:
                    # if self.match_number == 0:
                    #     print("CACA : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                    ship.task = None
                    continue

                if (
                    ship.target in unexplored_relics
                    and ship.energy > Global.UNIT_MOVE_COST * 5
                ):
                    relic_node_to_ship[ship.target] = ship
                else:
                    # if self.match_number == 0:
                    #     print("NOT ENOUGH ENERGY TO FIND REWARD : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
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
                # if self.match_number == 0:
                #     print("NO TARGET FOUND IN REWARD : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                return False

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]
                return True
            else:
                # if self.match_number == 0:
                #     print("NOT ACTION IN REWARD : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                return False
                

        can_pause = True
        for n, s in sorted(
            list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id
        ):
            if set_task(s, n, can_pause):
                if s.target == s.node:
                    # If one ship is stationary, we will move all the other ships.
                    # This will help generate more useful data in Global.REWARD_RESULTS.
                    can_pause = False
                    pass
            else:
                if s.task == "find_rewards":
                    # if self.match_number == 0:
                    #     print("COULDNT SET TASK IN REWARD : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
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

    def harvest(self):

        # if self.match_number == 0:
        #     print("WE ARE IN HARVEST", self.match_number*100+self.match_step, file=stderr)

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
                # if self.match_number == 0:
                #     print("NOT ACTION IN HARVEST : ", self.match_number*100+self.match_step, ship.unit_id, actions == None, ship.energy < energy, file=stderr)
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    # if self.match_number == 0:
                    #     print("SHOULD HARVEST BUT NO TARGET : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    # if self.match_number == 0:
                    #     print("TRIED TO HARVEST BUT NO : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            # if self.match_number == 0:
            #     print("HARVEST NO TARGETS WHATSOEVER", self.match_number*100+self.match_step, ship.unit_id, file=stderr)
            return

        for ship in self.fleet:
            # if self.match_number == 0:
            #     print("WTF LES REUFS : ", self.match_number*100+self.match_step, ship.unit_id, "task : ", ship.task, file=stderr)
            if ship.task:
                # if self.match_number == 0:
                #     print("HARVEST, NO TASK YET : ", self.match_number*100+self.match_step, ship.unit_id, "task : ", ship.task, file=stderr)
                continue

            # if self.match_number == 0:
            #     print("HARVEST, NO TASK YET : ", self.match_number*100+self.match_step, ship.unit_id, file=stderr)

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                # if self.match_number == 0:
                #     print("AFTER HARVEST NO TASK : ", target, self.match_number*100+self.match_step, ship.unit_id, file=stderr)
                ship.task = None
                ship.target = None

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
