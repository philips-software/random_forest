import asyncio

from mpyc.runtime import mpc

from src.best_split import select_best_attribute
from src.partition import partition_binary, partition_continuous
from src.tree import Branch, Leaf


@mpc.coroutine
async def train(samples, depth) -> asyncio.Future:
    print(f'Training at depth: {depth}')
    if depth > 0:
        attribute, threshold = select_best_attribute(samples)
        samples_left, samples_right = await partition(samples, attribute, threshold)
        left = await train(samples_left, depth=depth-1)
        right = await train(samples_right, depth=depth-1)
        label = samples.label(attribute)
        return Branch(label, threshold, left=left, right=right)
    else:
        pruned = mpc.is_zero(samples.len())
        outcome = samples.determine_class()
        return Leaf(outcome, pruned)


# reveals whether attribute is a continuous attribute
@mpc.coroutine
async def partition(samples, attribute, threshold) -> asyncio.Future:
    is_continuous = await mpc.output(samples.is_continuous(attribute))
    if is_continuous:
        return partition_continuous(samples, attribute, threshold)
    else:
        return partition_binary(samples, attribute)
