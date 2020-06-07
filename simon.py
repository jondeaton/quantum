
import jax.numpy as np
from quantum.states import *
from quantum.gates import *


def simon(n, f):
    """Simon's Algorithm.

    Given function

        f :: {0, 1} ^ n -> {0, 1} ^ n

    that is periodic on a, i.e.

        f(x) = f(x ⊕ a)

    find a.

    :param n: Number of Qbits.
    :param f: Function which takes as input a number in [0, 2^n) and outputs.
    :return: The period of f.
    """
    x = hadamard(n) @ zero(n) # Data register - top line.
    y = zero(n)
    uf = make_oracle(n, f)
    r = uf @ np.kron(x, y)
    s = np.kron(hadamard(n), eye(n)) @ r

    samples = [observe(s) for _ in range(n)]
    # TODO: solve system of linear equations mod 2.
