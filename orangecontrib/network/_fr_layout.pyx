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
import time

from libc.math cimport sqrt, fabs
from libc.stdlib cimport rand


ctypedef np.float64_t   double
ctypedef np.int32_t[:]  arr_i1_t
ctypedef double[:, :]   arr_f2_t
ctypedef double[:]      arr_f1_t


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


def fruchterman_reingold(arr_f1_t Edata,  # COO matrix constituents
						arr_i1_t Erow,   #
						arr_i1_t Ecol,   #
						double k,
						arr_f2_t pos,
						arr_i1_t fixed,
						int allowed_time,
						double sample_ratio,
                        double init_temp,
						callback,
						double callback_rate):
    cdef:
        double GRAVITY = 20
        arr_f2_t disp = np.empty((pos.shape[0], pos.shape[1]))
        arr_f1_t delta = np.empty(pos.shape[1])
        double mag, adj, weight, temp
        Py_ssize_t row, col, i, j, d, s
        Py_ssize_t n_nodes = pos.shape[0]
        Py_ssize_t n_edges = Edata.shape[0]
        Py_ssize_t n_dim = pos.shape[1]
        int have_callback = bool(callback)
        Py_ssize_t sample_size = max(int(round(n_nodes*sample_ratio)), 1)
        arr_i1_t sample = np.zeros(sample_size, dtype=np.int32)
        int callback_freq = int(round(1/callback_rate))

        int TEST_ITERATIONS = 10
        double last_perc = 0, perc
        int iterations = TEST_ITERATIONS
        int iteration = 0
        arr_f1_t temperature = np.linspace(init_temp, 0.01, TEST_ITERATIONS)
        double start_time = time.process_time()
        double elapsed_time

    with nogil:
        while iteration < iterations:
            disp[:, :] = 0
            for i in range(sample_size):
                sample[i] = rand() % n_nodes
            # Repulsive forces
            for i in range(n_nodes):
                for s in range(sample_size):
                    j = sample[s]
                    diff(pos, i, j, delta)
                    mag = magnitude(delta)
                    if mag == 0: continue
                    for d in range(n_dim):
                        disp[i, d] += delta[d] / mag * _Fr(k, mag)
                for d in range(n_dim):
                    disp[i,d] *= 1.0/sample_ratio
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
            temp = temperature[iteration]
            for i in range(n_nodes):
                mag = magnitude2(disp, i)
                if mag == 0: continue
                for d in range(n_dim):
                    pos[i, d] += disp[i, d] / mag * min(fabs(disp[i, d]), temp)

            # increase iteration count; if this is the end of test iterations,
            # set the actual number of iterations according to allowed time
            iteration += 1
            if iteration == TEST_ITERATIONS:
                with gil:
                    elapsed_time = time.process_time() - start_time
                    if elapsed_time < allowed_time:
                        iterations = min(
                            100 * n_nodes,
                            int(allowed_time / (elapsed_time / TEST_ITERATIONS))
                        )
                        temperature = np.linspace(0.1, 0.015, iterations) ** 2

            # Optionally call back with the new positions
            if iteration % callback_freq == 0 and have_callback:
                with gil:
                    perc = float(iteration) / iterations * 100
                    if not callback(np.asarray(pos), perc - last_perc):
                        break
                    last_perc = perc

    return pos
