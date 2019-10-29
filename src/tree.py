from dataclasses import dataclass
from typing import Any
from src.output import Secret, output


@dataclass
class Node(Secret):
    attribute: Any
    left: Any = None
    right: Any = None

    async def __output__(self):
        attribute = await output(self.attribute)
        left = await output(self.left) if self.left else self.left
        right = await output(self.right) if self.right else self.right
        return Node(attribute, left=left, right=right)
