from src.gini import gini_gain_quotient, avoid_zero
from src.maximum import index_of_maximum
from dataclasses import dataclass
from typing import Any


def select_best_attribute(samples):
    """
    Selects the best attribute for splitting a dataset.
    This is based on which attribute would yield the highest Gini gain.

    Keyword arguments:
    samples -- a list of Samples

    Return value:
    Column index of the attribute that is best suited for splitting the dataset.

    Attribute values and outcomes are expected to be either 0 or 1.
    """
    gains = calculate_gains(samples)
    gains = [(numerator, avoid_zero(denominator))
             for (numerator, denominator) in gains]
    return index_of_maximum(*gains)


def select_best_threshold(samples, column):
    gains = calculate_gains_for_thresholds(samples, column)
    gains = [(numerator, avoid_zero(denominator))
             for (numerator, denominator) in gains]
    index = index_of_maximum(*gains)
    return samples.column(column).getitem(index)


def calculate_gains(samples):
    number_of_attributes = samples.number_of_attributes

    gains = []
    for column in range(number_of_attributes):
        gain = calculate_gain_for_attribute(samples, column)
        gains.append(gain)

    return gains


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

    return aggregation.gini_gain_quotient()


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
