from mpyc.runtime import mpc
from src.dataset import ObliviousDataset


def partition_on(samples, is_active, attribute_index, threshold):
    selected_attribute = ObliviousDataset(*samples).column(attribute_index)

    left = [value <= threshold for value in selected_attribute]
    right = [(1 - l) for l in left]

    left = zero_if_inactive(left, is_active)
    right = zero_if_inactive(right, is_active)

    return left, right


def zero_if_inactive(values, is_active):
    return mpc.schur_prod(values, is_active)
