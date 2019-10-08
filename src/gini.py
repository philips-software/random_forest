def gini_impurity(amount_classified_as_zero, amount_classified_as_one):
    if amount_classified_as_one + amount_classified_as_zero == 0:
        return 1

    total = amount_classified_as_zero + amount_classified_as_one
    ratio_zero = amount_classified_as_zero / total
    ratio_one = amount_classified_as_one / total
    return 1 - ratio_zero ** 2 - ratio_one ** 2


def gini_gain_scaled_quotient(
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
