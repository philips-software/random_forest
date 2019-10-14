from mpyc.runtime import mpc


def partition_on(samples, is_active, attribute_index, threshold):
    selected_attribute = get_column(samples, attribute_index)

    left = [value <= threshold for value in selected_attribute]
    right = [(1 - l) for l in left]

    left = zero_if_inactive(left, is_active)
    right = zero_if_inactive(right, is_active)

    return left, right


def zero_if_inactive(values, is_active):
    return mpc.schur_prod(values, is_active)


def get_column(samples, attribute_index):
    num_attributes = len(samples[0])
    is_selected = [i == attribute_index for i in range(num_attributes)]
    return mpc.matrix_prod([is_selected], samples, True)[0]
