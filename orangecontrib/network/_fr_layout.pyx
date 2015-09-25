#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: embedsignature=True
#cython: infer_types=False
#cython: language_level=3

"""
Fruchterman, T. M. J., Reingold, E. M.,

Graph Drawing by Force-directed Placement.

http://emr.cs.iit.edu/~reingold/force-directed.pdf
"""

cimport numpy as np
import numpy as np

from libc.math cimport exp, log, sqrt, fabs


ctypedef np.float64_t   double
ctypedef np.int32_t[:]  arr_i1_t
ctypedef double[:, :]   arr_f2_t
ctypedef double[:]      arr_f1_t


def fruchterman_reingold_layout(G,
                                dim=2,
                                k=None,
                                pos=None,
                                fixed=None,
                                iterations=50,
                                weight='weight',
                                callback=None):
    """
    Position nodes using Fruchterman-Reingold force-directed algorithm.

    The parameters are equal to those of networkx.spring_layout(), with the
    following exceptions:

    Parameters
    ----------
    weight: str or np.ndarray
        The string attribute of the graph's edges that represents edge weights,
        or a 2D distance matrix.
    callaback: callable
        A function accepting `pos` ndarray to call after each iteration.
        The algorithm is stopped if the return value is ``False``-ish.

    """
    if G is None or len(G) == 0: return {}
    if len(G) == 1: return {G.nodes()[0]: [.5, .5]}

    # Determine initial positions
    pos_arr = np.random.random((len(G), dim))
    if isinstance(pos, dict):
        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    elif isinstance(pos, np.ndarray):
        pos_arr = pos
    fixed = np.array(fixed or [], dtype=np.int32)

    # Prepare edges info as sparse COO matrix parts
    nodelist = sorted(G)
    index = dict(zip(nodelist, range(len(nodelist))))
    try:
        if isinstance(weight, str):
            Erow, Ecol, Edata = zip(*[(index[u], index[v], w or 0)  # 0 if None
                                      for u, v, w in G.edges_iter(nodelist, data=weight)])
        elif isinstance(weight, np.ndarray):
            Erow, Ecol, Edata = zip(*[(index[u], index[v], weight[u, v])
                                      for u, v in G.edges_iter(nodelist)])
        else: raise TypeError('weight must be str or ndarray')
        # Don't allow zero weights
        try: min_w = np.min([i for i in Edata if i])
        except ValueError: min_w = 1
        Edata = np.clip(Edata, min_w, np.inf)
    except ValueError:  # No edges
        Erow, Ecol, Edata = [], [], []
    Erow = np.asarray(Erow, dtype=np.int32)
    Ecol = np.asarray(Ecol, dtype=np.int32)
    Edata = np.asarray(Edata, dtype=np.float64)
    # Optimal distance between nodes
    k = k or 1 / sqrt(pos_arr.shape[0])
    # Run...
    pos = np.asarray(_fruchterman_reingold(Edata, Erow, Ecol,
                                           k, pos_arr, fixed,
                                           iterations, callback))
    return pos


cdef inline void diff(arr_f2_t pos,
                      Py_ssize_t i,
                      Py_ssize_t j,
                      arr_f1_t out) nogil:
    cdef Py_ssize_t d
    for d in range(out.shape[0]):
        out[d] = pos[i, d] - pos[j, d]


cdef inline double magnitude(arr_f1_t a) nogil:
    cdef:
        double result = 0
        Py_ssize_t i
    for i in range(a.shape[0]):
        result += a[i] * a[i]
    return sqrt(result)


cdef inline double magnitude2(arr_f2_t a, Py_ssize_t i) nogil:
    cdef:
        double result = 0
        Py_ssize_t j
    for j in range(a.shape[1]):
        result += a[i, j] * a[i, j]
    return sqrt(result)


cdef inline double _Fr(double k, double x) nogil:
    return k*k / x


cdef inline double _Fa(double k, double x) nogil:
    return x*x / k


cdef arr_f2_t _fruchterman_reingold(arr_f1_t Edata,  # COO matrix constituents
                                    arr_i1_t Erow,   #
                                    arr_i1_t Ecol,   #
                                    double k,
                                    arr_f2_t pos,
                                    arr_i1_t fixed,
                                    int iterations,
                                    callback):
    cdef:
        double GRAVITY = 20
        arr_f1_t temperature = (.15 *
                                exp(log(10./1000) / iterations)**np.arange(iterations))
        arr_f2_t disp = np.empty((pos.shape[0], pos.shape[1]))
        arr_f1_t delta = np.empty(pos.shape[1])
        double mag, adj, weight, temp
        Py_ssize_t row, col, i, j, d, iteration
        Py_ssize_t n_nodes = pos.shape[0]
        Py_ssize_t n_edges = Edata.shape[0]
        Py_ssize_t n_dim = pos.shape[1]
        int have_callback = bool(callback)
    with nogil:
        temperature[0] = .8; temperature[1] = .5; temperature[2] = .3
        for iteration in range(iterations):
            temp = temperature[iteration]
            disp[:, :] = 0
            # Repulsive forces
            for i in range(n_nodes):
                for j in range(n_nodes):
                    diff(pos, i, j, delta)
                    mag = magnitude(delta)
                    if mag == 0: continue
                    for d in range(n_dim):
                        disp[i, d] += delta[d] / mag * _Fr(k, mag)
            # Attractive forces
            for i in range(n_edges):
                row, col, weight = Erow[i], Ecol[i], Edata[i]
                diff(pos, row, col, delta)
                mag = magnitude(delta)
                if mag == 0: continue
                for d in range(n_dim):
                    adj = delta[d] / mag * weight * _Fa(k, mag)
                    disp[row, d] -= adj
                    disp[col, d] += adj
            # Gravity; tend toward center
            for i in range(n_nodes):
                mag = magnitude2(pos, i)
                for d in range(n_dim):
                    disp[i, d] -= k * GRAVITY * mag * pos[i, d]
            # Keep fixed nodes fixed
            for i in range(fixed.shape[0]):
                i = fixed[i]
                for d in range(n_dim):
                    disp[i, d] = 0
            # Limit the maximum displacement
            for i in range(n_nodes):
                mag = magnitude2(disp, i)
                if mag == 0: continue
                for d in range(n_dim):
                    pos[i, d] += disp[i, d] / mag * min(fabs(disp[i, d]), temp)
            # Optionally call back with the new positions
            if have_callback:
                with gil:
                    if not callback(np.asarray(pos)):
                        break
            # If temperature too cool, finish early
            if temp < .0005:
                break
    return pos
