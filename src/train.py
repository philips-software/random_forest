from mpyc.runtime import mpc

from src.best_split import select_best_attribute
from src.partition import partition_binary, partition_continuous
from src.tree import Branch, Leaf


@mpc.coroutine
async def train(samples, depth):
    print(f'Training at depth: {depth}')
    if depth > 0:
        attribute, threshold = select_best_attribute(samples)
        samples_left, samples_right = partition(samples, attribute, threshold)
        left = train(samples_left, depth=depth-1)
        right = train(samples_right, depth=depth-1)
        return Branch(attribute, threshold, left=left, right=right)
    else:
        pruned = mpc.is_zero(samples.len())
        outcome = samples.determine_class()
        return Leaf(outcome, pruned)


# reveals whether attribute is a continuous attribute
@mpc.coroutine
async def partition(samples, attribute, threshold):
    is_continuous = await mpc.output(samples.is_continuous(attribute))
    if is_continuous:
        return partition_continuous(samples, attribute, threshold)
    else:
        return partition_binary(samples, attribute)
