from mpyc.runtime import mpc


def partition_on(samples, attribute_index, threshold):
    """
    Splits given data set into left and right part based on the
    threshold value of the attribute on which to split. Returns
    a pair containing the rows left of the split and those right
    of the split.

    Keyword arguments:
    samples -- ObliviousDataset
    attribute_index -- index of attribute (column) used for splitting
    threshold -- value that determines whether rows end up left or right

    Return value:
    (left, right) -- both ObliviousDatasets
    """
    selected_attribute = samples.column(attribute_index)

    left = [value <= threshold for value in selected_attribute]
    right = [(1 - l) for l in left]

    return samples.subset(left), samples.subset(right)


def zero_if_inactive(values, is_active):
    return mpc.schur_prod(values, is_active)
