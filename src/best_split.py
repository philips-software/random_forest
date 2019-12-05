from dataclasses import dataclass
from typing import Any

from src.array import ObliviousArray
from src.gini import avoid_zero, gini_gain_quotient
from src.maximum import maximum
from src.secint import secint as s


def select_best_attribute(samples):
    (gains, thresholds) = calculate_gains(samples)
    (_, index) = maximum(ObliviousArray.create(gains))
    threshold = ObliviousArray(thresholds).getitem(index)
    return (index, threshold)


def calculate_gains(samples):
    number_of_attributes = samples.number_of_attributes

    gains = []
    thresholds = []
    for column in range(number_of_attributes):
        if samples.is_continuous(column):
            (gain, threshold) = select_best_threshold(samples, column)
            gains.append(gain)
            thresholds.append(threshold)
        else:
            gain = calculate_gain_for_attribute(samples, column)
            gains.append(gain)
            thresholds.append(s(0))

    return gains, thresholds


def select_best_threshold(samples, column):
    gains = calculate_gains_for_thresholds(samples, column)
    (gain, index) = maximum(ObliviousArray.create(gains))
    threshold = samples.column(column).getitem(index)
    return (gain, threshold)


def calculate_gains_for_thresholds(samples, column):
    gains = []
    for threshold in samples.column(column):
        gain = calculate_gain_for_threshold(samples, column, threshold)
        gains.append(gain)

    return gains


def calculate_gain_for_attribute(samples, column):
    is_right = samples.column(column)
    return calculate_gain(is_right, samples.outcomes)


def calculate_gain_for_threshold(samples, column, threshold):
    is_right = samples \
        .column(column) \
        .map(lambda value: value > threshold)

    return calculate_gain(is_right, samples.outcomes)


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
