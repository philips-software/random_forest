from mpyc.runtime import mpc

from src.secint import secint as s


def index_of_maximum(*quotients):
    """
    Returns the index of the maximum in a list of quotients, aka "argmax".

    Only works for quotients that have positive numerator and denominator.
    """
    if not quotients:
        raise ValueError('expected at least one quotient')

    maximum = quotients[0]
    result = s(0)
    for index in range(1, len(quotients)):
        quotient = quotients[index]
        is_new_maximum = ge_quotient(quotient, maximum)
        result = mpc.if_else(is_new_maximum, index, result)
        maximum = mpc.if_else(is_new_maximum, list(quotient), list(maximum))
    return result


def ge_quotient(left, right):
    """
    Returns whether the left quotient is greater than or equal than the right
    quotient.

    Only works for quotients that have positive numerator and denominator.
    """
    (a, b) = left
    (c, d) = right
    return mpc.if_else(a * d >= b * c, True, False)
