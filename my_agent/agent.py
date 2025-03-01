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
            for x, y in nearby_positions(*node.coordinates, Global.RELIC_REWARD_RANGE):
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
        """
        Modified Fleet.update method to detect nebula energy reduction
        """
        previous_ship_states = {
            ship.unit_id: (ship.node, ship.energy)
            for ship in self.ships
            if ship.node is not None
        }

        # Update points from observation
        self.points = int(obs["team_points"][self.team_id])

        # Update ships with new positions and energy values
        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                previous_state = previous_ship_states.get(ship.unit_id)
                new_node = space.get_node(*position)

                # Store the new state
                ship.node = new_node
                ship.energy = int(energy)
                ship.action = None

                # Analyze energy reduction if we have previous state
                if previous_state and ship.node != previous_state[0]:
                    prev_node, prev_energy = previous_state

                    # Check if the ship moved to a nebula tile
                    if ship.node.type == NodeType.nebula:
                        # Calculate expected energy after movement
                        expected_energy = (
                            prev_energy - Global.UNIT_MOVE_COST + ship.node.energy
                        )

                        # Calculate actual nebula reduction
                        nebula_reduction = expected_energy - ship.energy

                        # Only record if reduction is positive (to avoid confusing with other energy changes)
                        if nebula_reduction in [0, 1, 2, 3, 5, 25]:
                            Global.NEBULA_ENERGY_OBSERVATIONS.append(nebula_reduction)

                            # Update the estimate once we have enough observations
                            if len(Global.NEBULA_ENERGY_OBSERVATIONS) >= 3:
                                # Use the most common value observed
                                from collections import Counter

                                counter = Counter(Global.NEBULA_ENERGY_OBSERVATIONS)
                                most_common = counter.most_common(1)[0][0]

                                # Only update if we're confident
                                if counter[most_common] >= 2:
                                    Global.NEBULA_ENERGY_REDUCTION = most_common
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

        self.find_relics()
        self.find_rewards()
        self.harvest()
        self.gather_energy()
        self.sap()
        self.check()
        return self.create_actions_array()

    def gather_energy(self):
        """
        Make ships gather energy before venturing into enemy territory.
        Ships targeting enemy reward nodes will first gather energy if they
        don't have at least 100 energy and are already in enemy territory.
        The gathering position should be within sap range of enemy reward nodes.
        """
        # Minimum energy threshold before venturing deep into enemy territory
        MIN_ENERGY_THRESHOLD = 150

        for ship in self.fleet:
            # Skip ships that are not targeting harvest or have no coordinates
            if (
                ship.task != "harvest"
                or ship.coordinates is None
                or ship.target is None
            ):
                continue

            # Check if the ship is targeting an enemy reward node
            target_in_enemy_territory = not is_team_sector(
                self.team_id, *ship.target.coordinates
            )

            # Check if the ship is already in enemy territory
            ship_in_enemy_territory = not is_team_sector(
                self.team_id, *ship.coordinates
            )

            # Only apply this strategy if ship is in enemy territory and targeting enemy reward
            if (
                ship_in_enemy_territory
                and target_in_enemy_territory
                and ship.energy < MIN_ENERGY_THRESHOLD
            ):
                # Look for high energy spots that are:
                # 1. Close to the ship's current position
                # 2. Within sap range of the target reward node
                # 3. Preferably in a safer position

                best_energy_spot = None
                best_energy_value = 0
                max_search_radius = (
                    5  # Limit search radius to avoid excessive computation
                )

                # Search for energy nodes around current position
                for x_offset in range(-max_search_radius, max_search_radius + 1):
                    for y_offset in range(-max_search_radius, max_search_radius + 1):
                        # Skip if search radius is too large (Manhattan distance)
                        if abs(x_offset) + abs(y_offset) > max_search_radius:
                            continue

                        # Calculate potential energy node position
                        pot_x, pot_y = warp_point(
                            ship.coordinates[0] + x_offset,
                            ship.coordinates[1] + y_offset,
                        )
                        pot_node = self.space.get_node(pot_x, pot_y)

                        # Skip if the node is not walkable or has no energy
                        if (
                            not pot_node.is_walkable
                            or pot_node.energy is None
                            or pot_node.energy <= 0
                        ):
                            continue

                        # Check if within sap range of the target
                        target_dist = max(
                            abs(pot_x - ship.target.coordinates[0]),
                            abs(pot_y - ship.target.coordinates[1]),
                        )

                        # Must be within sap range and have decent energy
                        if (
                            target_dist <= Global.UNIT_SAP_RANGE
                            and pot_node.energy > best_energy_value
                        ):
                            # Prefer spots in our territory if possible
                            safety_bonus = (
                                5 if is_team_sector(self.team_id, pot_x, pot_y) else 0
                            )

                            # Calculate total value considering energy and safety
                            total_value = pot_node.energy + safety_bonus

                            if total_value > best_energy_value:
                                best_energy_value = total_value
                                best_energy_spot = (pot_x, pot_y)

                # If we found a good energy spot, reroute the ship
                if best_energy_spot:
                    energy_node = self.space.get_node(*best_energy_spot)

                    # Calculate path to energy spot
                    path = astar(
                        create_weights(self.space), ship.coordinates, best_energy_spot
                    )

                    if path:
                        # Change the ship's immediate task to gather energy
                        actions = path_to_actions(path)
                        if actions:
                            ship.action = actions[0]
                            # Temporarily change task but keep the original target
                            ship.task = "gather_energy"
                            # print("GATHERING ", self.match_number*101+1+self.match_step, ship.unit_id, file=stderr)

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
            if ship.energy > Global.UNIT_MOVE_COST and ship.action == None:
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
                    action = self._find_best_single_action(ship, target)
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

    def sap(self):

        # Initial available ships
        available_ships = set()
        for ship in self.fleet:
            if ship.energy > Global.UNIT_SAP_COST and not (
                ship.task == "harvest" and ship.node != ship.target
            ):
                available_ships.add(ship)

        self.sap_1(available_ships)
        if available_ships:
            self.sap_2(available_ships)  # Sap other ships
        if available_ships and self.match_step >= 30:
            self.sap_3(available_ships)  # Blind sap on enemy reward nodes

    def sap_1(self, available_ships):
        # Find all enemy ship positions
        enemy_positions = [
            (enemy.coordinates[0], enemy.coordinates[1]) for enemy in self.opp_fleet
        ]

        # Find clusters of enemies (ships that are adjacent to each other)
        clusters = self._find_enemy_clusters(enemy_positions)

        # For each cluster, calculate best sap position
        for cluster in clusters:
            if len(cluster) < 2:  # Only care about clusters of 2+ ships
                continue

            # Calculate the optimal sap position for this cluster
            best_pos, damage_score = self._calculate_optimal_sap_position(cluster)

            if (
                best_pos and damage_score > 1.0
            ):  # Only worth it if we can hit more than one ship effectively
                # Find ships that can hit this position
                ships_in_range = []
                for ship in list(available_ships):
                    if ship.coordinates is None:
                        continue

                    sap_dist = max(
                        abs(best_pos[0] - ship.coordinates[0]),
                        abs(best_pos[1] - ship.coordinates[1]),
                    )

                    if sap_dist <= Global.UNIT_SAP_RANGE:
                        ships_in_range.append((ship, sap_dist))

                # Sort by distance (furthest first) to maximize range advantage
                ships_in_range.sort(key=lambda x: (-x[1], x[0].energy))

                # Use 1-2 ships depending on cluster size and damage potential
                ships_to_use = min(len(ships_in_range), 1 + (damage_score > 1.5))

                # Assign ships to sap this target
                for i in range(ships_to_use):
                    if i >= len(ships_in_range):
                        break

                    ship, _ = ships_in_range[i]
                    ship.action = ActionType.sap
                    ship.sap = best_pos
                    available_ships.remove(ship)

    def _find_enemy_clusters(self, enemy_positions):
        clusters = []
        visited = set()

        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = warp_point(x + dx, y + dy)
                neighbor_pos = (nx, ny)
                if neighbor_pos in enemy_positions:
                    neighbors.append(neighbor_pos)
            return neighbors

        def dfs(pos, current_cluster):
            visited.add(pos)
            current_cluster.append(pos)

            for neighbor in get_neighbors(pos):
                if neighbor not in visited:
                    dfs(neighbor, current_cluster)

        # Find all connected groups of enemy ships
        for pos in enemy_positions:
            if pos not in visited:
                current_cluster = []
                dfs(pos, current_cluster)
                clusters.append(current_cluster)

        return clusters

    def _calculate_optimal_sap_position(self, cluster):
        # Get all potential sap positions (all positions within UNIT_SAP_RANGE of any ship we have)
        potential_positions = set()

        # Get predicted positions for each enemy in the cluster
        predicted_positions = []
        for pos in cluster:
            # Find the enemy ship at this position
            enemy_ship = None
            for ship in self.opp_fleet:
                if ship.coordinates == pos:
                    enemy_ship = ship
                    break

            if enemy_ship:
                # Get the predicted position
                preshot_pos = self._preshot(enemy_ship)
                if preshot_pos:
                    predicted_positions.append(preshot_pos)
                else:
                    # If no prediction is available, use current position
                    predicted_positions.append(pos)
            else:
                # Fallback to current position if ship not found
                predicted_positions.append(pos)

        # Consider all positions in and around the predicted positions
        for pos in predicted_positions:
            x, y = pos
            # Center position
            potential_positions.add((x, y))

            # Adjacent positions
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = warp_point(x + dx, y + dy)
                    potential_positions.add((nx, ny))

        best_position = None
        best_score = 0

        # For each potential position, calculate the damage score using predicted positions
        for pos in potential_positions:
            score = 0
            for enemy_pos in predicted_positions:
                # Calculate Manhattan distance to determine splash damage
                dist = max(
                    abs(pos[0] - enemy_pos[0]),
                    abs(pos[1] - enemy_pos[1]),
                )

                # Direct hit
                if dist == 0:
                    score += 1.0
                # Splash damage (only if within 1 tile)
                elif dist == 1:
                    score += Global.UNIT_SAP_DROPOFF_FACTOR

            if score > best_score:
                best_score = score
                best_position = pos

        return best_position, best_score

    def sap_2(self, available_ships):

        if self.opp_fleet:
            prioritized_enemies = []
            normal_enemies = []

            for enemy_ship in self.opp_fleet:
                if enemy_ship.node.reward or self._is_near_relic(enemy_ship):
                    prioritized_enemies.append(enemy_ship)
                else:
                    normal_enemies.append(enemy_ship)

            # Process high-value targets first, then normal targets
            for i, target_list in enumerate([prioritized_enemies, normal_enemies]):
                for enemy_ship in target_list:
                    # Predict where the enemy will move to
                    predicted_pos = self._preshot(enemy_ship)
                    if predicted_pos:  # there is a preshot

                        # Sort available ships by distance to target (furthest first for range advantage)
                        ships_in_range = []
                        for ship in list(available_ships):
                            if ship.coordinates is None:
                                continue

                            sap_dist = max(
                                abs(predicted_pos[0] - ship.coordinates[0]),
                                abs(predicted_pos[1] - ship.coordinates[1]),
                            )

                            if sap_dist <= Global.UNIT_SAP_RANGE:
                                ships_in_range.append((ship, sap_dist))

                        # Sort by distance (furthest first) to maximize range advantage
                        ships_in_range.sort(key=lambda x: (-x[1], x[0].energy))

                        if (
                            i == 1 and len(ships_in_range) < 2
                        ):  # normal enemy ship and not many ships in range
                            continue
                        # Determine how many ships to use based on enemy value and our resources
                        # More aggressive approach: use more ships against high-value targets
                        if i == 0:
                            max_ships_to_use = 3 if enemy_ship.node.reward else 2
                        else:
                            max_ships_to_use = 2 if enemy_ship.node.reward else 1

                        ships_to_use = min(len(ships_in_range), max_ships_to_use)

                        # Assign ships to sap this target
                        for i in range(ships_to_use):
                            if i >= len(ships_in_range):
                                break

                            ship, _ = ships_in_range[i]
                            ship.action = ActionType.sap
                            ship.sap = predicted_pos
                            available_ships.remove(ship)

    def _preshot(self, ship):
        # Si l'enemy ship est sur un reward node, on considère qu'il est immobile
        x, y = ship.coordinates
        if ship.node.reward:
            # print("Preshot ",  (x,y), file=stderr)
            return ship.coordinates

        # Si l'ennemi est sur un asteroide, ne pas lui tirer dessus
        if ship.node.type == NodeType.asteroid:
            return None

        target, _ = find_closest_target(
            ship.coordinates,
            [n.coordinates for n in self.space.reward_nodes],
        )
        if not target:
            return None

        path = astar(create_weights(self.space), ship.coordinates, target)
        if not path:
            return None
        else:
            probable_coordinates = path[1]

        return probable_coordinates

    def sap_3(self, available_ships):
        reward_nodes = list(self.space.reward_nodes)

        if not reward_nodes:
            return

        invisible_reward_nodes = [node for node in reward_nodes if not node.is_visible]

        if not invisible_reward_nodes:
            return

        invisible_reward_nodes = list(
            filter(
                lambda node: not is_team_sector(self.team_id, node.x, node.y),
                invisible_reward_nodes,
            )
        )

        for ship in list(available_ships):
            if ship.coordinates is None or ship.energy <= Global.UNIT_SAP_COST:
                continue

            for target_node in invisible_reward_nodes:

                sap_dist = max(
                    abs(target_node.x - ship.coordinates[0]),
                    abs(target_node.y - ship.coordinates[1]),
                )

                if sap_dist <= Global.UNIT_SAP_RANGE and np.random.random() < 0.5:
                    ship.action = ActionType.sap
                    ship.sap = (target_node.x, target_node.y)
                    available_ships.remove(ship)
                    break

    def _get_sap_offset(self, ship):
        """Calculate the offset for sapping direction."""
        return ship.sap[0] - ship.coordinates[0], ship.sap[1] - ship.coordinates[1]

    def _is_near_relic(self, ship):
        """Check if a ship is near a relic (within reward range)."""
        if ship.coordinates is None:
            return False

        for relic_node in self.space.relic_nodes:
            x, y = relic_node.coordinates
            for xy in nearby_positions(x, y, Global.RELIC_REWARD_RANGE):
                if ship.coordinates == xy:
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

        return actions

    def find_relics(self):

        # Fix immobile ships after step 50 dans le premier match
        if (
            self.match_number == 0
            and all(node.explored_for_relic for node in self.space)
            and len(self.space.relic_nodes) == 0
        ):
            print("CLEAR MAP ", file=stderr)
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

        unexplored_relics = self.get_unexplored_relics()

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
        for n, s in sorted(
            list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id
        ):
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
