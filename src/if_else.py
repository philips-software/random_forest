from mpyc.runtime import mpc


def if_else(condition, if_true, if_false):
    if isinstance(if_true, tuple) and isinstance(if_false, tuple):
        return tuple(if_else(condition, list(if_true), list(if_false)))
    else:
        return mpc.if_else(condition, if_true, if_false)
