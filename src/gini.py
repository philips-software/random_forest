

def gini_gain_quotient(
        left_total,
        right_total,
        left_amount_classified_zero,
        left_amount_classified_one,
        right_amount_classified_zero,
        right_amount_classified_one):
    """
    Returns a pair of numbers that represent the Gini gain, given a split in
    the dataset.

    Keyword arguments:
    left_total -- the number of samples that on the left side of the split
    right_total -- the number of samples that on the right side of the split
    left_amount_classified_zero -- the number of samples on the left side that are classified as '0'
    left_amount_classified_one -- the number of samples on the left side that are classified as '1'
    right_amount_classified_zero -- the number of samples on the right side that are classified as '0'
    right_amount_classified_one -- the number of samples on the right side that are classified as '1'

    Return value:
    (numerator, denominator) -- such that the Gini gain equals
    (1 / total) * (numerator / denominator), where total is the total number
    of samples (left and right)

    May return a denominator of 0, use avoid_zero() to avoid division by zero.

    See also:
    Explanation of Gini gain -- https://victorzhou.com/blog/gini-impurity/
    Secure Training of Decision Trees with Continuous Attributes -- paper to be
    published by Mark Abspoel, Daniel Escudero and Nikolaj Volgushev
    """
    numerator = \
        right_total * (left_amount_classified_zero ** 2 +
                       left_amount_classified_one ** 2) + \
        left_total * (right_amount_classified_zero ** 2 +
                      right_amount_classified_one ** 2)
    denominator = left_total * right_total
    return (numerator, denominator)


def avoid_zero(denominator, precision=8):
    """
    Avoids division by zero by scaling the denominator.

    Keyword arguments:
    denominator -- the denominator to be scaled
    precision -- (defaults to 8), the amount of scaling that is performed

    Return value:
    denominator * precision + 1
    """
    return denominator * precision + 1
