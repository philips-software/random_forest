from dataclasses import dataclass
from typing import Any

from src.array import ObliviousArray
from src.gini import avoid_zero, gini_gain_quotient
from src.maximum import maximum
from src.secint import secint as s
from src.sort import sort


def select_best_attribute(samples):
    (gains, thresholds) = calculate_gains(samples)
    (_, index) = maximum(ObliviousArray.create(gains))
    threshold = ObliviousArray(thresholds).getitem(index)
    return (index, threshold)


def calculate_gains(samples):
    number_of_attributes = samples.number_of_attributes

    gains = []
    thresholds = []
    outcomes = samples.outcomes
    for attribute in range(number_of_attributes):
        column = samples.column(attribute)
        if samples.is_continuous(attribute):
            s_column, s_outcomes = sort(column, outcomes)
            (gain, threshold) = select_best_threshold(s_column, s_outcomes)
            gains.append(gain)
            thresholds.append(threshold)
        else:
            gain = calculate_gain_for_attribute(column, outcomes)
            gains.append(gain)
            thresholds.append(s(0))

    return gains, thresholds


def select_best_threshold(column, outcomes):
    gains = calculate_gains_for_thresholds(column, outcomes)
    (gain, index) = maximum(gains)
    threshold = column.getitem(index)
    return (gain, threshold)


def calculate_gains_for_thresholds(column, outcomes):
    gains = column.map(lambda _: None)
    is_right = column.map(lambda _: s(1))
    for index in range(len(column.values)):
        is_right.values[index] = s(0)
        gains.values[index] = calculate_gain(is_right, outcomes)
    return gains


def calculate_gain_for_attribute(column, outcomes):
    return calculate_gain(column, outcomes)


def calculate_gain(is_right, outcomes):
    aggregation = Aggregation(total=is_right.len())

    aggregation.right_total = is_right.sum()

    right_classified_one = is_right.select(outcomes)
    left_classified_one = right_classified_one.map(lambda value: 1 - value)

    aggregation.right_amount_classified_one = right_classified_one.sum()
    aggregation.left_amount_classified_one = left_classified_one.sum()

    (numerator, denominator) = aggregation.gini_gain_quotient()
    return (numerator, avoid_zero(denominator))


@dataclass
class Aggregation:
    total: Any = 0
    right_total: Any = 0
    left_amount_classified_one: Any = 0
    right_amount_classified_one: Any = 0

    @property
    def left_total(self):
        return self.total - self.right_total

    @property
    def left_amount_classified_zero(self):
        return self.left_total - self.left_amount_classified_one

    @property
    def right_amount_classified_zero(self):
        return self.right_total - self.right_amount_classified_one

    def gini_gain_quotient(self):
        return gini_gain_quotient(
            self.left_total,
            self.right_total,
            self.left_amount_classified_zero,
            self.left_amount_classified_one,
            self.right_amount_classified_zero,
            self.right_amount_classified_one
        )
