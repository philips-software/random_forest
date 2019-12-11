from src.array import ObliviousArray, ObliviousSelection
from src.secint import secint as s


def sort(column, outcomes):
    if isinstance(column, ObliviousArray):
        return sort_array(column, outcomes)
    else:
        return sort_selection(column, outcomes)


def sort_array(column, outcomes):
    values, outcome_values = sort_lists(
        column.values,
        outcomes.values
    )
    return (
        ObliviousArray(values),
        ObliviousArray(outcome_values)
    )


def sort_selection(column, outcomes):
    values, outcome_values, included = sort_lists(
        column.values,
        outcomes.values,
        column.included
    )
    return (
        ObliviousSelection(values, included),
        ObliviousSelection(outcome_values, included)
    )


def sort_lists(*lists):
    result = [l.copy() for l in lists]
    bsort(result)
    return tuple(result)


# Sorting network implementation from
# https://github.com/lschoe/mpyc/blob/master/demos/SecureSortingNetsExplained.ipynb


def bsort(xs):
    def bitonic_sort(lo, n, up=True):
        if n > 1:
            m = n // 2
            bitonic_sort(lo, m, not up)
            bitonic_sort(lo + m, n - m, up)
            bitonic_merge(lo, n, up)

    def bitonic_merge(lo, n, up):
        if n > 1:
            # set m as the greatest power of 2 less than n:
            m = 2**((n - 1).bit_length() - 1)
            for i in range(lo, lo + n - m):
                bitonic_compare(i, i + m, up)
            bitonic_merge(lo, m, up)
            bitonic_merge(lo + m, n - m, up)

    def bitonic_compare(i, j, up):
        b = (xs[0][i] > xs[0][j]) ^ ~s(up)
        for x in xs:
            d = b * (x[j] - x[i])
            x[i], x[j] = x[i] + d, x[j] - d

    bitonic_sort(0, len(xs[0]))
    return xs
