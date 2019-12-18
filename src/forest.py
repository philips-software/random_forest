from math import sqrt

from mpyc.runtime import mpc

from src.dataset import ObliviousDataset, Sample
from src.output import output
from src.secint import secint
from src.train import train
from src.array import ObliviousArray


async def train_forest(samples, amount, depth, amount_of_features=None):
    if not amount_of_features:
        amount_of_features = int(sqrt(samples.number_of_attributes))

    forest = []
    for _ in range(amount):
        selection = await random_attributes(samples, amount_of_features)
        selection = bootstrap(selection)
        tree = await train(selection, depth)
        forest.append(tree)
    return forest


def bootstrap(samples):
    selected_samples = [samples.choice() for _ in range(len(samples))]
    return ObliviousDataset.create(
        *selected_samples,
        continuous=samples.continuous,
        labels=samples.labels
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
    return ObliviousDataset.create(
        *smaller_samples,
        continuous=continuous,
        labels=labels
    )


# reveals which chosen columns are continuous
async def random_columns(samples, amount):
    indices = range(samples.number_of_attributes)
    selected = mpc.random.sample(secint, indices, amount)
    return (
        [samples.column(index) for index in selected],
        [await output(samples.is_continuous(index)) for index in selected],
        [samples.label(index) for index in selected]
    )
