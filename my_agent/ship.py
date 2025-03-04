from sys import stderr

from node import Node
from base import ActionType

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
            f"Ship({self.unit_id}, node={self.node.coordinates if self.node else None}, energy={self.energy})"
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