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

from libc.math cimport sqrt, fabs


ctypedef np.float64_t   double
ctypedef np.int32_t[:]  arr_i1_t
ctypedef double[:, :]   arr_f2_t
ctypedef double[:]      arr_f1_t
ctypedef np.int8_t[:]   arr_b1_t

cdef:
    double GRAVITY = 20


cdef inline double sqr(double x) nogil:
    return x * x


# disp = d / |d| * F_r(d), where d is the vector between the two points and
# F_r(d) = k^2 / |d|.
# Thus dist = d * k^2 / |d|^2. We work with |d| and avoid square roots.
cpdef void repulse(arr_f2_t pos, double k, arr_f2_t disp) nogil:
    cdef:
        Py_ssize_t n_nodes = pos.shape[0]
        Py_ssize_t n_dim = pos.shape[1]
        double dx, dy, f, d2
        Py_ssize_t i, j

        double k2 = sqr(k)

    for i in range(n_nodes):
        for j in range(i):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            d2 = sqr(dx) + sqr(dy)
            if d2 == 0:
                continue
            f = k2 / d2
            disp[i, 0] += dx * f
            disp[i, 1] += dy * f
            disp[j, 0] -= dx * f
            disp[j, 1] -= dy * f


cpdef void repulse_sampled(arr_f2_t pos, arr_i1_t sample,
                      double k,
                      arr_f2_t disp) nogil:
    cdef:
        Py_ssize_t n_nodes = pos.shape[0]
        Py_ssize_t n_dim = pos.shape[1]
        double dx, dy, f, d2
        Py_ssize_t i, j, s

        Py_ssize_t sample_size = sample.shape[0]
        Py_ssize_t inv_sample_ratio = n_nodes / sample_size
        double k2 = sqr(k)

    for i in range(n_nodes):
        for s in range(sample_size):
            j = sample[s]
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            d2 = sqr(dx) + sqr(dy)
            if d2 == 0:
                continue
            f = k2 / d2 * inv_sample_ratio
            disp[i, 0] += dx * f
            disp[i, 1] += dy * f


cpdef void attract(arr_f2_t pos,
                   arr_i1_t edge_src,  arr_i1_t edge_dst, arr_f1_t edge_weights,
                   double k,
                   arr_f2_t disp) nogil:
    cdef:
        Py_ssize_t n_edges = edge_src.shape[0]

        double dx, dy, f, mag2, weight
        Py_ssize_t i, j, e

        double inv_k = 1 / k
        int weighted = edge_weights.shape[0] != 0

    # TODO are weights in any way normalized?!
    for e in range(n_edges):
        i, j = edge_src[e], edge_dst[e],
        dx = pos[i][0] - pos[j][0]
        dy = pos[i][1] - pos[j][1]
        mag2 = sqr(dx) + sqr(dy)
        f = sqrt(mag2) * inv_k
        if weighted:
            f *= edge_weights[e]
        dx *= f
        dy *= f
        disp[i, 0] -= dx
        disp[i, 1] -= dy
        disp[j, 0] += dx
        disp[j, 1] += dy


cpdef void gravity(arr_f2_t pos, double k, arr_f2_t disp) nogil:
    cdef:
        double dx, dy, f, mag, weight
        Py_ssize_t i

    for i in range(pos.shape[0]):
        """
        Circular gravity; gravity along axes (below) seems to work better
        f = k * GRAVITY * sqrt(sqr(pos[i, 0]) + sqr(pos[i, 1]))
        disp[i, 0] -= f * pos[i, 0]
        disp[i, 1] -= f * pos[i, 1]
        """
        f = k * GRAVITY
        disp[i, 0] -= f * fabs(pos[i, 0]) * pos[i, 0]
        disp[i, 1] -= f * fabs(pos[i, 1]) * pos[i, 1]


cpdef void move(arr_f2_t pos, arr_f2_t disp, double temp) nogil:
    cdef:
        double mag, f
        Py_ssize_t i

        double temp2 = sqr(temp)

    for i in range(pos.shape[0]):
        mag = sqr(disp[i, 0]) + sqr(disp[i, 1])
        if mag == 0:
            continue
        if mag > temp2:
            f = temp / sqrt(mag)
            pos[i, 0] += disp[i, 0] * f
            pos[i, 1] += disp[i, 1] * f
        else:
            pos[i, 0] += disp[i, 0]
            pos[i, 1] += disp[i, 1]


def fruchterman_reingold_step(arr_f2_t pos,
                              arr_i1_t sample,
                              arr_i1_t edge_src,
                              arr_i1_t edge_dst,
                              arr_f1_t edge_weights,
                              double k,
                              double temp,
                              # preallocated for efficiency; same shape as pos
                              arr_f2_t disp):
    cdef:
        double mag, adj, weight
        Py_ssize_t row, col, i, j, d, s
        Py_ssize_t n_nodes = pos.shape[0]
        Py_ssize_t n_dim = pos.shape[1]

        Py_ssize_t sample_size = sample.shape[0]
        Py_ssize_t inv_sample_ratio = n_nodes / sample_size

    with nogil:
        disp[:, :] = 0
        attract(pos, edge_src, edge_dst, edge_weights, k, disp)
        if  sample.shape[0] == 0:
            repulse(pos, k, disp)
        else:
            repulse_sampled(pos, sample, k, disp)

        gravity(pos, k, disp)
        move(pos, disp, temp)
