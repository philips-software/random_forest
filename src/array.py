import operator
from dataclasses import dataclass
from typing import Any, Sequence

from mpyc.runtime import mpc

from src.output import Secret, output
from src.secint import secint as s


@dataclass(frozen=True)
class ObliviousArray(Secret):
    values: [Any]
    included: [Any]

    @classmethod
    def create(cls, *values, included=None):
        if len(values) == 1 and isinstance(values[0], Sequence):
            values = values[0]
        else:
            values = list(values)
        return cls(values, included)

    def len(self):
        if self.included:
            return mpc.sum(self.included)
        else:
            return len(self.values)

    def select(self, *include):
        if len(include) == 1 and isinstance(include[0], (Sequence, ObliviousArray)):
            include = include[0]
        else:
            include = list(include)

        if isinstance(include, ObliviousArray):
            return self.select(*include.included_values_or_zero())

        if self.included == None:
            return type(self)(self.values, included=include)

        include = mpc.schur_prod(self.included, include)
        return type(self)(self.values, included=include)

    def included_values_or_zero(self):
        if self.included:
            return mpc.schur_prod(self.values, self.included)
        else:
            return self.values

    def map(self, function):
        values = list(map(function, self.values))
        return ObliviousArray(values, self.included)

    def sum(self):
        included_values = self.included_values_or_zero()
        theSum = mpc.sum(included_values)
        return theSum

    def __getitem__(self, index):
        is_selected = [i == index for i in range(len(self.values))]
        return mpc.matrix_prod([is_selected], [self.values], True)[0][0]

    async def __output__(self):
        values = [await output(value) for value in self.values]
        if self.included:
            included = await output(self.included)
        else:
            included = [True] * len(self.values)
        return [values[i] for i in range(len(values)) if included[i]]
