def gini_impurity(amount_classified_as_zero, amount_classified_as_one):
    if amount_classified_as_one + amount_classified_as_zero == 0:
        return 1

    total = amount_classified_as_zero + amount_classified_as_one
    ratio_left = amount_classified_as_zero / total
    ratio_right = amount_classified_as_one / total
    return 1 - ratio_left ** 2 - ratio_right ** 2
