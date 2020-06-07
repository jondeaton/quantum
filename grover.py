import quantum
from quantum.gates import *
from quantum.states import *


def make_oracle(n, f):
    """Constructs an oracle that flips the sign of one CBS.

    :param n: Number of Qbits.
    :param f: Function which takes i in [0, 2^n) and outputs True only
    for one single value in its domain, otherwise outputs False.
    :return: Function which takes as inputs Quantum state, and outputs
    that state with only the CBS flipped for which "f" returns True.
    """
    flipper = np.array([1 if f(i) else -1 for i in range(2 ** n)])
    return lambda x: x * flipper


def grover(n, oracle):
    """Grover's Algorithm.

    Example:
        n = 8 -- number of Qbits
        oracle = make_oracle(n, lambda i: i == 42)
        state = grover(n, oracle)
        sample(state)

    :param n: Number of Qbits.
    :param oracle: Funciton which takes as input a quantum state and produces a quantum
    state with a single
    :return: Quantum stats that can be sampled to retrieve CBS which oracle flips.
    """
    phi = hadamard(n) @ zero(n)
    diffusion = np.outer(2 * phi, phi) - eye(n)

    r = int(np.pi * np.sqrt(2 ** n) / 4)
    for _ in range(r):
        phi = diffusion @ oracle(phi)
    return phi
