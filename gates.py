
import jax.numpy as np
import functools

from quantum.states import *

dtype = np.float16

z = np.array([1, 0], dtype=dtype)
o = np.array([0, 1], dtype=dtype)
i = 1j

# X / QNOT / Pauli X / Bit Flip Gate
quot = np.outer(o, z) + np.outer(z, o)


@functools.lru_cache()
def hadamard(n):
    """Hadamard operator for m qbits.

    :param n: Number of qbits
    :return: Matrix operator of size 2^m
    """
    if n < 0:
        raise ValueError(f'Invalid number of dimensions: {n}')
    if n == 0:
        return np.array(1, dtype=dtype)
    if n == 1:
        return np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2)
    else:
        return np.kron(hadamard(1), hadamard(n - 1))


# Y / Bit-and-Phase Flip / Pauli Y Gate
pauli_y = np.array([[0, -i], [i, 0]], dtype=dtype)


def phase_shift(theta):
    """Phase shift gate.

    :param theta: Phase shift angle.
    :return: Matrix operator for 2 qbits.
    """
    return np.outer(z, z) + np.outer(np.exp(i * theta) * o, o)


# Z / Phase Flip / Pauli Z Gate
pauli_x = phase_shift(np.pi)


def eye(n):
    """Identity gate."""
    return np.eye(2 ** n, dtype=dtype)


def make_oracle(n, f):
    """

    Form Quantum Oracle Matrix.
    Creates the unitary matrix representing an oracle from arbitrary function

        f : {0, 1} ^ n -> {0, 1} ^ n

    The matrix Uf, acts on (i.e. takes as input) two n-qbit registers. Its action
    on the first and second registers, when fully concentrated in the two computational
    basis states ∣x⟩∣y⟩ is given by

        Uf∣x⟩∣y⟩ = ∣x⟩∣y ⊕ f(x)⟩

    That is, the first register remain unchanged and the second register becomes
    the bitwise xor of its previous CBS (i.e. y) with the function, f, applied to the
    CBS of the first register, (i.e. x). The gate extends linearly to register states
    that are not fully concentrated in any CBS.

    As described by A Michael Loceff, Course in Quantum Computing (page 358)
    http://lapastillaroja.net/wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf

    We form this by letting the columns of the matrix Uf be the outputs for each
    computational basis state for the two registers.

    :param f: Callable function mapping {0, 1} ^ n -> {0, 1} ^ n
    :param n: Number of Qbits.
    :return: Unitary matrix that acts on two n-Qbit registers.
    """
    return np.array([
        [
            np.kron(cbs(n, x), cbs(n, y | f(x)))
            for x in range(2 ** n)
        ] for y in range(2 ** n)], dtype=dtype).T


def qft(n):
    """Quantum Fourrier Transform.

    :param n: Number of Qbits in input/output state.
    :return: Unitary matrix representation of QFT gate.
    """
    m = 2 ** n
    powers = np.outer(np.arange(m, dtype=dtype), np.arange(m, dtype=dtype))
    w = np.exp(2 * np.pi * i / m)  # m'th root of unity.
    return np.power(w, - powers) / np.sqrt(m)
