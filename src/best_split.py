from src.gini import gini_gain_quotient, avoid_zero
from src.maximum import index_of_maximum
from mpyc.runtime import mpc
from dataclasses import dataclass, field
from typing import Any


def select_best_attribute(samples):
    """
    Selects the best attribute for splitting a dataset.
    This is based on which attribute would yield the highest Gini gain.

    Keyword arguments:
    samples -- a matrix containing a sample in each row and an attribute value in each cell
    outcomes -- an array that for each sample contains the expected outcome

    Return value:
    Column index of the attribute that is best suited for splitting the dataset.

    Attribute values and outcomes are expected to be either 0 or 1.
    """
    gains = calculate_gains(samples)
    gains = [(numerator, avoid_zero(denominator))
             for (numerator, denominator) in gains]
    return index_of_maximum(*gains)


def calculate_gains(samples):
    if len(samples) == 0:
        raise ValueError("Expected at least one sample")

    number_of_attributes = len(samples[0].inputs)

    gains = []
    for column in range(number_of_attributes):
        gain = calculate_gain_for_attribute(samples, column)
        gains.append(gain)

    return gains


def calculate_gain_for_attribute(samples, column):
    number_of_samples = len(samples)
    aggregation = Aggregation(total=number_of_samples)
    for row in range(number_of_samples):
        is_active = samples.is_active(row)
        value = samples[row][column]
        outcome = samples[row].outcome
        aggregation.left_amount_classified_one += (
            (1 - value) * outcome) * is_active
        aggregation.right_amount_classified_one += (
            value * outcome) * is_active

    aggregation.right_total = samples.column(column).sum()
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
