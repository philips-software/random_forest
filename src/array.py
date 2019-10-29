from dataclasses import dataclass
from typing import Any, Sequence
from mpyc.runtime import mpc
from src.output import Secret, output
from functools import reduce
import operator

s = mpc.SecInt()


@dataclass
class ObliviousArray(Secret):
    values: [Any]
    included: [Any]

    def __init__(self, *values, included=None):
        if len(values) == 1 and isinstance(values[0], Sequence):
            values = values[0]
        self.values = values
        self.included = included

    def len(self):
        if self.included:
            return reduce(operator.add, self.included, s(0))
        else:
            return len(self.values)

    def select(self, *include):
        if len(include) == 1 and isinstance(include[0], (Sequence, ObliviousArray)):
            include = include[0]

        if isinstance(include, ObliviousArray):
            return self.select(*include.included_values_or_zero())

        if self.included == None:
            return type(self)(self.values, included=include)

        include = mpc.schur_prod(list(self.included), include)
        return type(self)(self.values, included=include)

    def included_values_or_zero(self):
        if self.included:
            return mpc.schur_prod(list(self.values), self.included)
        else:
            return self.values

    def map(self, function):
        values = map(function, self.values)
        return ObliviousArray(*values, included=self.included)

    def reduce(self, neutral_element, operation):
        values = self.values
        included = self.included
        if included:
            values = [mpc.if_else(included[i], values[i], neutral_element)
                      for i in range(len(self.values))]
        return reduce(operation, values, neutral_element)

    def sum(self):
        return self.reduce(0, operator.add)

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
