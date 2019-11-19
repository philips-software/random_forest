from dataclasses import dataclass
from typing import Any
from src.output import Secret, output
from src.secint import secint as s


@dataclass
class Branch(Secret):
    attribute: Any
    left: Any = None
    right: Any = None

    def pretty_print(self, leader=''):
        next_leader = leader + '|  '
        if self.left:
            print(f"{leader}if attr_{self.attribute} == 0")
            self.left.pretty_print(next_leader)
        if self.right:
            print(f"{leader}if {self.attribute} == 1")
            self.right.pretty_print(next_leader)

    async def __output__(self):
        attribute = await output(self.attribute)
        left = await output(self.left) if self.left else self.left
        right = await output(self.right) if self.right else self.right
        return Branch(attribute, left=left, right=right)


@dataclass
class Leaf(Secret):
    outcome_class: Any
    pruned: Any = s(False)

    async def __output__(self):
        outcome_class = await output(self.outcome_class)
        pruned = await output(self.pruned)
        return Leaf(outcome_class, pruned)
