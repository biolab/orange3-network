from functools import wraps

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import row_norms

from orangecontrib.network import Network

__all__ = ["path", "cycle", "complete", "complete_bipartite", "barbell",
           "ladder", "circular_ladder", "grid", "hypercube",
           "lollipop", "star", "geometric"]

def from_csr(f):
    @wraps(f)
    def wrapped(*args):
        edges = f(*args)
        return Network(
            range(edges.shape[0]), edges,
            name=f"{f.__name__}{args}".replace(",)", ")"))
    return wrapped


def from_row_col(f):
    @wraps(f)
    def wrapped(*args):
        row, col, *n = f(*args)
        n = n[0] if n else max(np.max(row), np.max(col)) + 1
        edges = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        return Network(
            range(n), edges,
            name=f"{f.__name__}{args}".replace(",)", ")"))
    return wrapped


def from_dense(f):
    @wraps(f)
    def wrapped(*args):
        m = f(*args)
        return Network(
            range(len(m)), sp.csr_matrix(m),
            name=f"{f.__name__}{args}".replace(",)", ")"))
    return wrapped


@from_row_col
def path(n):
    return np.arange(n - 1), np.arange(n - 1) + 1, n


@from_row_col
def cycle(n):
    return np.arange(n), np.arange(1, n + 1) % n


@from_csr
def complete(n):
    if n < 5_000:
        return sp.triu(np.ones((n, n)), k=1, format="csr")

    # manually construct sparse matrix elements to avoid storing a dense (n x n) matrix in memory
    indptr = [0]
    for n_above_diag in range(n - 1, 0, -1):
        indptr.append(indptr[-1] + n_above_diag)
    indptr.append(indptr[-1])
    indptr = np.array(indptr)
    indices = np.concatenate([np.arange(row, n) for row in range(1, n)])
    data = np.ones(n * (n - 1) // 2)

    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


@from_row_col
def complete_bipartite(n, m):
    return np.repeat(np.arange(n), m), n + np.arange(n * m) % m


@from_dense
def barbell(n, m):
    e = np.zeros((n + m, n + m))
    e[:n, :n] = 1  # complete
    e[-m:, -m:] = 1  # complete
    e[n, -m - 1] = e[n - 1, -m] = 1  # bridge
    e -= np.eye(n + m, n + m)  # no loops
    return np.triu(e)


def ladder(n):
    return grid(n, 2)


def circular_ladder(n):
    return grid(n, 2, True)


@from_csr
def grid(n, m=4, circular=False):
    t = n * m
    row = np.arange(2 * t - n) % t
    col = np.zeros(2 * t - n, dtype=int)
    data = np.ones(2 * t - n)

    col[:t] = np.arange(1, t + 1)  # horizontal edges
    col[n - 1:t:n] = np.arange(m)  # circle to prevent index out of range
    if not circular:
        data[n - 1:t:n] = 0  # eliminate circle

    col[t:] = np.arange(n, t)  # vertical edges
    edges = sp.csr_matrix((data, (row, col)), shape=(t, t))
    edges.eliminate_zeros()
    return edges


# https://math.stackexchange.com/questions/2328139/adjacency-matrix-for-n-dimensional-hypercube-graph
def _hypercube(ndim):
    """Recursively construct the edge-connectivity of a hypercube

    Parameters
    ----------
    ndim : int
        Dimension of the hypercube

    Returns
    -------
    ndarray, [2**ndim, 2**ndim], bool
        connectivity pattern of the hypercube
    """
    if ndim == 0:
        return np.array([[0]])
    else:
        d = _hypercube(ndim-1)
        i = np.eye(len(d))
        return np.block([
            [d, i],
            [i, d],
        ])


@from_dense
def hypercube(ndim):
    return _hypercube(ndim)


@from_row_col
def star(n):
    return np.zeros(n), np.arange(1, n + 1)


@from_dense
def lollipop(n, m):
    e = np.zeros((n + m, n + m))
    e[:n, :n] = 1 - np.eye(n)
    e[n, -m - 1] = e[n - 1, -m] = 1  # bridge
    for i in range(m, n + m):
        e[i - 1, i] = 1
    return np.triu(e)


def geometric(n_nodes, n_edges):
    n_pairs = n_nodes * (n_nodes - 1) // 2
    if n_edges > n_pairs:
        raise ValueError(
            f"There are only {n_pairs} (< {n_edges}) possible edges between " 
            f"{n_nodes} points")
    xy = np.random.random((n_nodes, 2))
    xx = row_norms(xy, squared=True)[:, np.newaxis]
    distances = np.dot(xy, xy.T)
    distances *= -2
    distances += xx
    distances += xx.T
    ur = np.triu_indices(n_nodes, k=1)
    # skip zeros and repetitions
    dist_threshold = np.partition(distances[ur], n_edges)[n_edges]
    mask = distances <= dist_threshold
    mask[np.tril_indices(n_nodes)] = False
    row, col = mask.nonzero()
    edges = sp.csr_matrix((np.ones(len(row)), (row, col)),
                          shape=(n_nodes, n_nodes))
    return Network(
        range(n_nodes), edges,
        name=f"geometric({n_nodes},{n_edges})",
        coordinates=xy
    )
