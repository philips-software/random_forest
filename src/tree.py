from dataclasses import dataclass
from typing import Any
from src.output import Secret, output


@dataclass(frozen=True)
class Branch(Secret):
    attribute: Any
    threshold: Any
    left: Any = None
    right: Any = None

    def pretty_print(self, leader=''):
        next_leader = leader + '|  '
        if self.left:
            print(f"{leader}if attr_{self.attribute} <= {self.threshold}")
            self.left.pretty_print(next_leader)
        if self.right:
            print(f"{leader}if attr_{self.attribute} > {self.threshold}")
            self.right.pretty_print(next_leader)

    async def __output__(self):
        attribute = await output(self.attribute)
        threshold = await output(self.threshold)
        left = await output(self.left) if self.left else self.left
        right = await output(self.right) if self.right else self.right
        return Branch(attribute, threshold, left, right)


@dataclass(frozen=True)
class Leaf(Secret):
    outcome: Any
    pruned: Any

    def pretty_print(self, leader=''):
        if self.pruned == 1:
            print(f"{leader}<pruned>")
        else:
            print(f"{leader}outcome: {self.outcome}")

    async def __output__(self):
        outcome = await output(self.outcome)
        pruned = await output(self.pruned)
        return Leaf(outcome, pruned)
