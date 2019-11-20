from src.dataset import ObliviousDataset, Sample
from src.secint import secint as s


def sample(ins, out):
    return Sample([s(i) for i in ins], s(out))


samples = ObliviousDataset.create(
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
