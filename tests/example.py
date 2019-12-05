from src.dataset import ObliviousDataset, Sample
from src.secint import secint as s


def sample(ins, out):
    return Sample([s(i) for i in ins], s(out))


binary_samples = ObliviousDataset.create(
    sample([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], 0),
    sample([1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], 1),
    sample([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], 0),
    sample([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 1),
    sample([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 0),
    sample([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1], 1),
    sample([1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0], 1),
    sample([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 1),
    sample([1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0], 1),
    sample([1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0], 0),
    sample([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0], 1),
    sample([1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], 0),
    sample([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0], 1),
    sample([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 1)
)

continuous_samples = ObliviousDataset.create(
    sample([1, 1, 1, 2], 1),
    sample([1, 1, 1, 3], 1),
    sample([1, 1, 1, 4], 1),
    sample([1, 1, 3, 1], 1),
    sample([1, 1, 3, 2], 1),
    sample([1, 1, 3, 3], 1),
    sample([1, 1, 3, 4], 1),
    sample([1, 1, 3, 5], 1),
    sample([1, 1, 4, 1], 1),
    sample([3, 2, 5, 5], 1),
    sample([3, 3, 1, 1], 0),
    sample([3, 3, 1, 2], 0),
    sample([3, 3, 2, 1], 0),
    sample([3, 3, 2, 2], 0),
    sample([3, 3, 2, 3], 0),
    sample([3, 3, 2, 4], 0),
    sample([3, 3, 2, 5], 1),
    sample([3, 3, 3, 1], 0),
    continuous_attributes=[0, 1, 2, 3]
)
