import abc
from mpyc.runtime import mpc


async def output(value):
    if isinstance(value, Secret):
        return await value.output()
    return await mpc.output(value)


class Secret(abc.ABC):

    @abc.abstractmethod
    async def output(self):
        pass
