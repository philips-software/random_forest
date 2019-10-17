import abc
from collections import Sequence
from mpyc.runtime import mpc


async def output(value):
    if isinstance(value, Secret):
        return await value.output()
    elif isinstance(value, Sequence) and not isinstance(value, list):
        return await output(list(value))
    return await mpc.output(value)


class Secret(abc.ABC):

    @abc.abstractmethod
    async def output(self):
        pass
