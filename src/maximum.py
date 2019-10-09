def index_of_maximum(*quotients):
    """
    Returns the index of the maximum in a list of quotients.

    Only works for quotients that have positive numerator and denominator.
    """
    if not quotients:
        raise ValueError('expected at least one quotient')

    maximum = quotients[0]
    result = 0
    for index in range(1, len(quotients)):
        quotient = quotients[index]
        is_new_maximum = ge_quotient(quotient, maximum)
        result = if_else(is_new_maximum, index, result)
        maximum = (
            if_else(is_new_maximum, quotient[0], maximum[0]),
            if_else(is_new_maximum, quotient[1], maximum[1]))
    return result


def ge_quotient(left, right):
    """
    Returns whether the left quotient is greater than or equal than the right
    quotient.

    Only works for quotients that have positive numerator and denominator.
    """
    (a, b) = left
    (c, d) = right
    return if_else(a * d >= b * c, True, False)


def if_else(condition, when_true, when_false):
    return condition * (when_true - when_false) + when_false
