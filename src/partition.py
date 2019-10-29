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

    left = selected_attribute.map(lambda value: value <= threshold)
    right = left.map(lambda value: 1 - value)

    return samples.select(left), samples.select(right)


def partition_on_binary_attribute(samples, attribute_index):
    selected_attribute = samples.column(attribute_index)

    right = selected_attribute
    left = right.map(lambda value: 1 - value)

    return samples.select(left), samples.select(right)
