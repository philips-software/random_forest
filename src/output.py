import abc
from collections.abc import Sequence
from mpyc.runtime import mpc


async def output(value):
    if isinstance(value, Secret):
        return await value.__output__()
    elif isinstance(value, Sequence) and not isinstance(value, list):
        return await output(list(value))
    return await mpc.output(value)


class Secret(abc.ABC):

    @abc.abstractmethod
    async def __output__(self):
        pass
