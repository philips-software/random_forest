from functools import reduce

from mpyc.runtime import mpc

from src.secint import secint as s


def maximum(*quotients):
    """
    Returns both the maximum quotient and the index of the maximum in a list.

    Only works for quotients that have positive numerator and denominator.
    """
    if not quotients:
        raise ValueError('expected at least one quotient')

    def max(previous, current):
        if not previous:
            return (current, s(0), 0)

        (maximum, index_of_maximum, index) = previous
        index += 1

        is_new_maximum = ge_quotient(current, maximum)
        index_of_maximum = mpc.if_else(is_new_maximum, index, index_of_maximum)
        maximum = tuple(mpc.if_else(is_new_maximum,
                                    list(current),
                                    list(maximum)))

        return (maximum, index_of_maximum, index)

    maximum, index_of_maximum, _ = reduce(max, quotients, None)
    return maximum, index_of_maximum


def ge_quotient(left, right):
    """
    Returns whether the left quotient is greater than or equal than the right
    quotient.

    Only works for quotients that have positive numerator and denominator.
    """
    (a, b) = left
    (c, d) = right
    return a * d >= b * c
