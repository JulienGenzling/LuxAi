from sys import stderr

import copy
import numpy as np
from scipy.signal import convolve2d

from node import Node
from base import (
    Global,
    SPACE_SIZE,
    get_match_step,
    get_match_number,
    get_opposite,
    NodeType,
    warp_point,
)
from pathfinding import nearby_positions


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
        match = get_match_number(step)
        match_step = get_match_step(step)

        new_relic = False

        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                # We have found a new relic.
                self._update_relic_status(*xy, status=True)
                for x, y in nearby_positions(*xy, Global.RELIC_REWARD_RANGE):
                    if not self.get_node(x, y).reward:
                        self._update_reward_status(x, y, status=None)
                new_relic = True

        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)

            if not node.explored_for_relic:
                all_relics_found = False

            if not node.explored_for_reward:
                # print(f"[match {match}| step {match_step}] Node", node, "is not explored for reward", file=stderr)
                all_rewards_found = False

        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found
        # print(f"[match {match}| step {match_step}] All rewards found", Global.ALL_REWARDS_FOUND, file=stderr)

        num_relics_th = 2 * min(match, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1

        if new_relic:
            Global.ALL_RELICS_FOUND = False
            Global.REWARD_RESULTS = []
            # print(len(Global.REWARD_RESULTS), Global.ALL_RELICS_FOUND, Global.ALL_REWARDS_FOUND, len(self._relic_nodes), num_relics_th, file=stderr)

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
                # print(f"[match {match}| step {match_step}] Update rewards, {len(self._relic_nodes)} relics, theoric number: {num_relics_th}", file=stderr)
                self._update_reward_status_from_relics_distribution()
                self._update_reward_results(obs, team_id, team_reward)
                self._update_reward_status_from_reward_results()

        # if new_relic:
        #     show_exploration_map(self)

    def _update_reward_status_from_reward_results(self):
        """
        A more robust approach to updating the reward-status for nodes based on
        Global.REWARD_RESULTS. We first run the old logic, then do an additional pass
        comparing sets of unknown tiles across different results to deduce if certain
        tiles are guaranteed to be (or not be) reward tiles.
        """

        # We'll store any newly deduced statuses and apply them after inference
        must_be_reward = set()   # set of (x,y) coords that must be reward
        must_be_empty = set()    # set of (x,y) coords that must be empty

        # --- PASS 1: The existing simple logic -----------------------------------
        for result in Global.REWARD_RESULTS:
            unknown_nodes = set()
            known_reward_count = 0

            for n in result["nodes"]:
                # skip nodes that are known empty (explored_for_reward but not reward)
                if n.explored_for_reward and not n.reward:
                    continue

                # skip nodes that are known reward
                if n.reward:
                    known_reward_count += 1
                    continue

                # otherwise, it's an unknown node
                unknown_nodes.add(n)

            if not unknown_nodes:
                continue  # all fully resolved for this result

            reward_delta = result["reward"] - known_reward_count

            if reward_delta == 0:
                # none of these unknown nodes gave any points
                for node in unknown_nodes:
                    must_be_empty.add(node.coordinates)

            elif reward_delta == len(unknown_nodes):
                # all of these unknown nodes gave points
                for node in unknown_nodes:
                    must_be_reward.add(node.coordinates)

            # else:  0 < reward_delta < len(unknown_nodes)
            #        can't decide with single-step logic -> handle in pass 2

        # Immediately apply the certain inferences from PASS 1
        for (x, y) in must_be_empty:
            self._update_reward_status(x, y, status=False)
        for (x, y) in must_be_reward:
            self._update_reward_status(x, y, status=True)

        
        # --- PASS 2: Attempt more advanced inference across pairs of results -----
        changed = True
        while changed:
            changed = False

            new_must_empty = set()
            new_must_reward = set()

            # We'll do pairwise comparisons
            for i in range(len(Global.REWARD_RESULTS)):
                for j in range(i + 1, len(Global.REWARD_RESULTS)):

                    resA = Global.REWARD_RESULTS[i]
                    resB = Global.REWARD_RESULTS[j]

                    # Step 1: Gather unknown nodes for each result ignoring
                    # nodes that are definitely known from must_be_reward/empty
                    A_unknown, A_known = self.collect_unknown_and_known_reward(
                        resA, must_be_empty, must_be_reward
                    )
                    B_unknown, B_known = self.collect_unknown_and_known_reward(
                        resB, must_be_empty, must_be_reward
                    )

                    # Step 2: Effective new reward that belongs to these unknown sets
                    A_reward = resA["reward"] - A_known
                    B_reward = resB["reward"] - B_known

                    # Step 3: Compare sets
                    diffA = A_unknown - B_unknown  # the set difference
                    diffB = B_unknown - A_unknown

                    # If there's exactly 1 node difference in sets:
                    if len(diffA) == 1 and len(diffB) == 0:
                        (diff_node,) = diffA  # unpack the single node
                        # Compare reward differences
                        if A_reward == B_reward + 1:
                            # diff_node must be a reward
                            if diff_node not in must_be_reward:
                                new_must_reward.add(diff_node)
                        elif A_reward == B_reward:
                            # diff_node must be empty
                            if diff_node not in must_be_empty:
                                new_must_empty.add(diff_node)

                    elif len(diffB) == 1 and len(diffA) == 0:
                        (diff_node,) = diffB
                        if B_reward == A_reward + 1:
                            if diff_node not in must_be_reward:
                                new_must_reward.add(diff_node)
                        elif B_reward == A_reward:
                            if diff_node not in must_be_empty:
                                new_must_empty.add(diff_node)

                    # You can further extend logic for 2+ differences, etc.

            # Apply any new deductions we found in this pass
            if new_must_empty or new_must_reward:
                changed = True
                for coords in new_must_empty:
                    if coords not in must_be_empty and coords not in must_be_reward:
                        must_be_empty.add(coords)
                        self._update_reward_status(*coords, status=False)

                for coords in new_must_reward:
                    if coords not in must_be_reward and coords not in must_be_empty:
                        must_be_reward.add(coords)
                        self._update_reward_status(*coords, status=True)


    def collect_unknown_and_known_reward(self, result, must_be_empty, must_be_reward):
        """
        Return:
        - unknown_set: set of (x, y) coords still unknown in this result
        - known_count: how many nodes in this result are deduced to be reward
        """
        unknown_set = set()
        known_count = 0

        for node in result["nodes"]:
            coords = node.coordinates

            if coords in must_be_empty:
                # definitely empty, skip
                continue
            elif coords in must_be_reward:
                # definitely reward, increment known reward
                known_count += 1
            else:
                # If node.explored_for_reward but not node.reward => also empty
                if node.explored_for_reward and not node.reward:
                    # same as must_be_empty
                    continue
                # Otherwise, it's still unknown
                unknown_set.add(coords)

        return unknown_set, known_count

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
                # print(Global.OBSTACLE_MOVEMENT_DIRECTION, Global.OBSTACLE_MOVEMENT_PERIOD, file=stderr)

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
        if len(obstacles_movement_status) < 21:
            return

        num_movements = sum(obstacles_movement_status)

        if num_movements <= 0:
            return 40
        elif num_movements <= 1:
            return 20
        elif num_movements <= 2:
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
