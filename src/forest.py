from math import sqrt

from mpyc.runtime import mpc

from src.dataset import ObliviousDataset, Sample
from src.output import output
from src.secint import secint
from src.train import train


async def train_forest(samples, amount, depth, amount_of_features=None):
    if not amount_of_features:
        amount_of_features = int(sqrt(samples.number_of_attributes))

    forest = []
    for _ in range(amount):
        selection = await random_attributes(samples, amount_of_features)
        await mpc.barrier()
        selection = bootstrap(selection)
        await mpc.barrier()
        tree = train(selection, depth)
        forest.append(tree)
    return forest


def bootstrap(samples):
    return ObliviousDataset(
        [samples.choice() for _ in range(len(samples))],
        samples.continuous
    )


async def random_attributes(samples, amount):
    columns, continuous, labels = await random_columns(samples, amount)
    outcomes = samples.outcomes
    smaller_samples = []
    for r in range(len(samples)):
        inputs = []
        outcome = outcomes[r]
        for c in range(len(columns)):
            inputs.append(columns[c][r])
        smaller_samples.append(Sample(inputs, outcome))
    return ObliviousDataset(smaller_samples, continuous, labels)


# reveals which chosen columns are continuous
async def random_columns(samples, amount):
    indices = range(samples.number_of_attributes)
    selected = mpc.random.sample(secint, indices, amount)
    return (
        [samples.column(index) for index in selected],
        [await output(samples.is_continuous(index)) for index in selected],
        [samples.label(index) for index in selected]
    )
