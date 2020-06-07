
import jax
import jax.numpy as np
import random

dtype=np.float16

def cbs(n, i):
    """Computational basis state.

    Produces a vector representation of a quantum state of n Qbits with all of the
    mass concentrated in computational basis state i.

    :param n: Number of Qbits.
    :param i: Index of state to concentrate mass in.
    :return: Vector representation of CBS of index i.
    """
    return np.array(np.arange(2 ** n) == i, dtype=dtype)


def zero(n):
    """Computational basis state of all zeros.


    :param n: Number of Qbits.
    :return: Vector representation of CBS of n Qbits all set to zero.
    """
    return cbs(n, 0)


def one(n):
    """Computation basis state of all ones.

    :param n: Number of Qbits.
    :return: Vector representation of CBS of n Qbits all set to one.
    """
    return cbs(n, 2 ** n - 1)


def observe(state, seed=None):
    """Observes the classical state of a quantum system.

    Collapses the quantum state into a classical state, sampling
    from the distribution of classical states given by the amplitudes
    in the quantum system described by state.

    :param state: Vector representation of quantum state to observe.
    :param seed: Optional seed with which to sample randomly.
    :return: Integer in [0, state.shape[0])
    """
    seed = seed if seed is not None else random.randint(0, np.iinfo(np.int32).max)
    key = jax.random.PRNGKey(seed)
    p = np.real(np.conj(state) * state)
    return jax.random.categorical(key, np.log(p))
