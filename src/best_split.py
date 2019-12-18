from dataclasses import dataclass
from typing import Any

from mpyc.runtime import mpc

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
    outcomes = samples.outcomes
    for attribute in range(number_of_attributes):
        column = samples.column(attribute)
        if samples.is_continuous(attribute):
            s_column = samples.sorted_column(attribute)
            s_outcomes = samples.sorted_outcomes(attribute)
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
    is_right = column.map(lambda _: s(0))
    selection = [None for _ in range(len(column.values))]
    last_considered_value = s(-1)
    for index in reversed(range(len(column.values))):
        gains.values[index] = calculate_gain(is_right, outcomes)
        is_right.values[index] = s(1)
        is_duplicate = column.values[index] == last_considered_value
        selection[index] = ~ is_duplicate
        last_considered_value = mpc.if_else(
            column.is_included(index),
            column.values[index],
            last_considered_value
        )
    return gains.select(selection)


def calculate_gain_for_attribute(column, outcomes):
    return calculate_gain(column, outcomes)


def calculate_gain(is_right, outcomes):
    aggregation = Aggregation(total=outcomes.len())

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
