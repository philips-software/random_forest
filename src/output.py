import abc
from collections.abc import Sequence
from mpyc.runtime import mpc


async def output(value):
    if isinstance(value, Secret):
        return await value.__output__()
    elif isinstance(value, Sequence):
        return [await output(x) for x in value]
    return await mpc.output(value)


class Secret(abc.ABC):

    @abc.abstractmethod
    async def __output__(self):
        pass
