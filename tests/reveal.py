from mpyc.runtime import mpc
from src.output import output


def reveal(secret):
    return mpc.run(output(secret))
