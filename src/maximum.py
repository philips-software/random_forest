from src.if_else import if_else
from src.secint import secint as s


def maximum(quotients):
    """
    Returns both the maximum quotient and the index of the maximum in an
    oblivious sequence.

    Only works for quotients that have positive numerator and denominator.
    """

    def max(previous, current):
        (maximum, index_of_maximum, index) = previous

        is_new_maximum = ge_quotient(current, maximum)
        index_of_maximum = if_else(is_new_maximum, index, index_of_maximum)
        maximum = tuple(if_else(is_new_maximum,
                                list(current),
                                list(maximum)))

        return (maximum, index_of_maximum, index + 1)

    neutral = (s(0), s(0))
    initial = (neutral, s(0), s(0))
    maximum, index_of_maximum, _ = quotients.reduce(neutral, max, initial)
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
