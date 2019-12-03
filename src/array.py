from dataclasses import dataclass
from typing import Any, Sequence

from mpyc.runtime import mpc

from src.output import Secret, output
from src.secint import secint
from src.sequence import ObliviousSequence


@dataclass(frozen=True)
class ObliviousArray(Secret, ObliviousSequence):
    values: [Any]

    @classmethod
    def create(cls, *values):
        if len(values) == 1 and isinstance(values[0], Sequence):
            values = list(values[0])
        else:
            values = list(values)
        return cls(values)

    def __len__(self):
        """length of this dataset as a plain number"""
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def getitem(self, index):
        """get item with a secret index"""
        length = len(self.values)
        included = mpc.unit_vector(index, length)
        return mpc.sum(mpc.schur_prod(self.values, included))

    def len(self):
        """length of this dataset as a secure number"""
        return secint(len(self))

    def select(self, *include):
        if len(include) == 1 and isinstance(include[0], (Sequence, ObliviousSequence)):
            include = include[0]
        else:
            include = list(include)

        if isinstance(include, ObliviousArray):
            return self.select(*include.values)

        if isinstance(include, ObliviousSelection):
            return self.select(*include.included_values_or_zero())

        return ObliviousSelection(self.values, included=include)

    def map(self, function):
        return ObliviousArray(list(map(function, self.values)))

    def sum(self):
        return mpc.sum(self.values)

    async def __output__(self):
        return [await output(value) for value in self.values]


@dataclass(frozen=True)
class ObliviousSelection(Secret, ObliviousSequence):
    values: [Any]
    included: [Any]

    def len(self):
        return mpc.sum(self.included)

    def map(self, function):
        mapped = ObliviousArray(self.values).map(function)
        return ObliviousSelection(mapped.values, self.included)

    def included_values_or_zero(self):
        return mpc.schur_prod(self.values, self.included)

    def sum(self):
        return mpc.sum(self.included_values_or_zero())

    def select(self, *include):
        selection = ObliviousArray(self.values).select(*include)
        included = mpc.schur_prod(self.included, selection.included)
        return ObliviousSelection(selection.values, included)

    async def __output__(self):
        values = await output(ObliviousArray(self.values))
        included = await output(self.included)
        return [values[i] for i in range(len(values)) if included[i]]
