from mpyc.runtime import mpc

from src.secint import secint as s


def maximum(*quotients):
    """
    Returns both the maximum quotient and the index of the maximum in a list.

    Only works for quotients that have positive numerator and denominator.
    """
    if not quotients:
        raise ValueError('expected at least one quotient')

    maximum = quotients[0]
    index_of_maximum = s(0)
    for index in range(1, len(quotients)):
        quotient = quotients[index]
        is_new_maximum = ge_quotient(quotient, maximum)
        index_of_maximum = mpc.if_else(is_new_maximum, index, index_of_maximum)
        maximum = tuple(mpc.if_else(is_new_maximum,
                                    list(quotient),
                                    list(maximum)))
    return (maximum, index_of_maximum)


def ge_quotient(left, right):
    """
    Returns whether the left quotient is greater than or equal than the right
    quotient.

    Only works for quotients that have positive numerator and denominator.
    """
    (a, b) = left
    (c, d) = right
    return a * d >= b * c
