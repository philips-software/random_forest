import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any

from mpyc.random import random_unit_vector
from mpyc.runtime import mpc
from mpyc.sectypes import Share

from src.array import ObliviousArray
from src.output import Secret, output
from src.secint import secint
from src.sequence import ObliviousSequence


@dataclass
class Sample(Secret):
    inputs: [Any]
    outcome: Any

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]

    def __add__(self, other):
        return Sample(
            mpc.vector_add(self.inputs, other.inputs),
            self.outcome + other.outcome
        )

    def __mul__(self, other):
        multiplications = mpc.scalar_mul(other, self.inputs + [self.outcome])
        multiplied_outcome = multiplications.pop()  # mutates multiplications
        return Sample(multiplications, multiplied_outcome)

    async def __output__(self):
        return Sample(await output(self.inputs),
                      await output(self.outcome))


@dataclass(frozen=True)
class __Dataset__(ObliviousSequence):
    samples: ObliviousArray
    number_of_attributes: int

    def len(self):
        return self.samples.len()

    def map(self, function):
        return self.samples.map(function)

    def sum(self):
        return self.samples.sum()

    @property
    def outcomes(self):
        return self.map(lambda sample: sample.outcome)

    def determine_class(self):
        return self.outcomes.sum() * 2 > self.len()

    def column(self, index):
        if isinstance(index, Share):
            is_selected = mpc.unit_vector(index, self.number_of_attributes)
            return self.map(
                lambda sample:
                ObliviousArray
                .create(sample.inputs)
                .select(is_selected)
                .sum()
            )
        else:
            return self.map(lambda sample: sample.inputs[index])

    def select(self, *include):
        selection = self.samples.select(*include)
        return ObliviousDatasetSelection(selection, self.number_of_attributes)

    async def __output__(self):
        return await output(self.samples)


class ObliviousDataset(__Dataset__, Secret):
    def __init__(self, values):
        number_of_attributes = len(values[0]) if len(values) > 0 else 0
        samples = ObliviousArray.create(values)
        __Dataset__.__init__(self, samples, number_of_attributes)

    @classmethod
    def create(cls, *values):
        return ObliviousDataset(values)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def choice(self):
        length = len(self.samples)
        included = random_unit_vector(secint, length)
        selected = [self.samples[i] * included[i] for i in range(length)]
        return reduce(operator.add, selected)


class ObliviousDatasetSelection(__Dataset__, Secret):
    pass
