from mpyc.runtime import mpc
from src.secint import secint
from src.dataset import ObliviousDataset, Sample
from src.train import train


def train_forest(samples, amount, depth):
    return [train(bootstrap(samples), depth) for _ in range(amount)]


def bootstrap(samples):
    return ObliviousDataset([samples.choice() for _ in range(len(samples))])


def random_attributes(samples, amount):
    columns = random_columns(samples, amount)
    outcomes = samples.outcomes
    smaller_samples = []
    for r in range(len(samples)):
        inputs = []
        outcome = outcomes[r]
        for c in range(len(columns)):
            inputs.append(columns[c][r])
        smaller_samples.append(Sample(inputs, outcome))
    return ObliviousDataset(smaller_samples)


def random_columns(samples, amount):
    indices = range(samples.number_of_attributes)
    selected = mpc.random.sample(secint, indices, amount)
    return [samples.column(index) for index in selected]
