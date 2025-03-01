from sys import stderr

from ship import Ship
from base import Global, NodeType
from space import Space

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