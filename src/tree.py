from dataclasses import dataclass
from typing import Any
from src.output import Secret, output


@dataclass
class Node(Secret):
    attribute: Any

    def __init__(self, attribute):
        self.attribute = attribute

    async def __output__(self):
        return Node(await output(self.attribute))
