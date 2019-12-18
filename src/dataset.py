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
from src.sort import sort


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
class ObliviousDatasetSelection(ObliviousSequence, Secret):
    samples: ObliviousSequence
    number_of_attributes: int
    continuous: [bool]
    labels: [int]

    def len(self):
        return self.samples.len()

    def map(self, function):
        return self.samples.map(function)

    def reduce(self, neutral_element, operation):
        return self.samples.reduce(neutral_element, operation)

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
        selected = self.samples.select(*include)
        return ObliviousDatasetSelection(
            selected,
            self.number_of_attributes,
            self.continuous,
            self.labels
        )

    def is_included(self, index):
        return self.samples.is_included(index)

    def is_continuous(self, attribute_index):
        if isinstance(attribute_index, Share):
            continuous = ObliviousArray(list(map(secint, self.continuous)))
            return continuous.getitem(attribute_index)
        else:
            return self.continuous[attribute_index]

    def label(self, attribute_index):
        labels = ObliviousArray(self.labels)
        return labels.getitem(attribute_index)

    async def __output__(self):
        return await output(self.samples)


class ObliviousDataset(ObliviousDatasetSelection):
    @classmethod
    def create(cls, *samples, continuous=None, labels=None):
        samples = ObliviousArray.create(samples)
        number_of_attributes = len(samples[0]) if len(samples) > 0 else 0
        if not continuous:
            continuous = [False for i in range(number_of_attributes)]
        if not labels:
            labels = [secint(i) for i in range(number_of_attributes)]
        return ObliviousDataset(
            samples,
            number_of_attributes,
            continuous,
            labels
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def choice(self):
        length = len(self.samples)
        included = random_unit_vector(secint, length)
        selected = [self.samples[i] * included[i] for i in range(length)]
        return reduce(operator.add, selected)

    def sort(self):
        sorted_columns = []
        sorted_outcomes_list = []
        sorted_indices_list = []
        for index in range(self.number_of_attributes):
            if self.is_continuous(index):
                sorted_column, sorted_outcomes, sorted_indices = sort(
                    self.column(index),
                    self.outcomes
                )
                sorted_columns.append(sorted_column)
                sorted_outcomes_list.append(sorted_outcomes)
                sorted_indices_list.append(sorted_indices)
            else:
                sorted_columns.append(None)
                sorted_outcomes_list.append(None)
                sorted_indices_list.append(None)
        return ObliviousSortedDataset(
            self.samples,
            self.number_of_attributes,
            self.continuous,
            self.labels,
            sorted_columns,
            sorted_outcomes_list,
            sorted_indices_list
        )


@dataclass(frozen=True)
class ObliviousSortedSelection(ObliviousDatasetSelection):
    sorted_columns: [ObliviousSequence]
    sorted_outcomes_list: [ObliviousSequence]
    sorted_indices_list: [[Share]]

    def assert_column_is_sorted(self, index):
        if not self.is_continuous(index):
            raise IndexError(
                f"column {index} is not sorted, " +
                "perhaps it is not a continuous attribute?"
            )

    def sorted_column(self, index):
        self.assert_column_is_sorted(index)
        return self.sorted_columns[index]

    def sorted_outcomes(self, index):
        self.assert_column_is_sorted(index)
        return self.sorted_outcomes_list[index]

    def select(self, *include):
        selected = super().select(*include)
        sorted_includes = []
        for index in range(self.number_of_attributes):
            sorted_indices = ObliviousArray(self.sorted_indices_list[index])
            included = ObliviousArray(selected.samples.included)
            sorted_included = sorted_indices \
                .map(lambda index: included.getitem(index))
            sorted_includes.append(sorted_included)
        selected_sorted_columns = [self.sorted_columns[i].select(
            sorted_includes[i]) for i in range(self.number_of_attributes)]
        selected_sorted_outcomes_list = [self.sorted_outcomes_list[i].select(
            sorted_includes[i]) for i in range(self.number_of_attributes)]
        return ObliviousSortedSelection(
            selected.samples,
            selected.number_of_attributes,
            selected.continuous,
            selected.labels,
            selected_sorted_columns,
            selected_sorted_outcomes_list,
            sorted_includes
        )


@dataclass(frozen=True)
class ObliviousSortedDataset(ObliviousDataset, ObliviousSortedSelection):
    pass
