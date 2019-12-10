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


def compare_and_swap(a, b, *lists):
    c = lists[0][a] > lists[0][b]
    for x in lists:
        d = c * (x[b] - x[a])
        x[a], x[b] = x[a] + d, x[b] - d

# Sorting network implementation from
# https://github.com/lschoe/mpyc/blob/master/demos/SecureSortingNetsExplained.ipynb


def oddeven_merge(lo, hi, r):
    step = r * 2
    if step < hi - lo:
        yield from oddeven_merge(lo, hi, step)
        yield from oddeven_merge(lo + r, hi, step)
        yield from [(i, i + r) for i in range(lo + r, hi - r, step)]
    else:
        yield (lo, lo + r)


def oddeven_merge_sort_range(lo, hi):
    """ sort the part of x with indices between lo and hi.

    Note: endpoints (lo and hi) are included.
    """
    if (hi - lo) >= 1:
        # if there is more than one element, split the input
        # down the middle and first sort the first and second
        # half, followed by merging them.
        mid = lo + ((hi - lo) // 2)
        yield from oddeven_merge_sort_range(lo, mid)
        yield from oddeven_merge_sort_range(mid + 1, hi)
        yield from oddeven_merge(lo, hi, 1)


def oddeven_merge_sort(length):
    """ "length" is the length of the list to be sorted.
    Returns a list of pairs of indices starting with 0 """
    yield from oddeven_merge_sort_range(0, length - 1)


def bsort(*xs):
    lists = xs[0]

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
        b = (lists[0][i] > lists[0][j]) ^ ~s(up)
        for x in lists:
            d = b * (x[j] - x[i])
            x[i], x[j] = x[i] + d, x[j] - d

    bitonic_sort(0, len(lists[0]))
    return lists
