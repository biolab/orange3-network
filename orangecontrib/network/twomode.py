from collections import namedtuple

import numpy as np
import scipy.sparse as sp

from orangecontrib import network


NoWeights, WeightConnections, WeightWeightedConnections,\
    WeightGeo, WeightGeoDeg, WeightInput, WeightOutput, WeightMin, WeightMax \
    = range(9)


def to_single_mode(net, mode_mask, conn_mask, weighting):
    """
    Convert two-mode network into a single mode

    Args:
        net: network to convert
        mode_mask (boolean array): a mask with nodes to connect
        conn_mask (boolean array): a mask with nodes to use for connecting
        weighting (int): normalization for edge weigthts

    Returns:
        single-mode network
    """
    new_net = network.Graph()
    new_net.add_nodes_from(range(mode_mask.sum()))
    mode_edges = _filtered_edges(net, mode_mask, conn_mask, weighting > 1)
    if mode_edges is not None:
        new_edges = Weighting[weighting].func(mode_edges)
        new_edges = new_edges.tocoo()
        new_net.add_weighted_edges_from(
            zip(new_edges.row, new_edges.col, new_edges.data))
    return new_net


def _normalize(a, *nominators):
    for nominator in nominators:
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.reciprocal(nominator)
        inv[nominator == 0] = 1
        a = a.multiply(inv)
    return a


def _dot_edges(normalization):
    def norm_dot(edges):
        edges = normalization(
            edges, wuu=lambda: edges.sum(axis=0), wvv=lambda: edges.sum(axis=1))
        new_edges = np.dot(edges, edges.T)
        new_edges.setdiag(0)
        new_edges.eliminate_zeros()
        return new_edges
    return norm_dot


def _weight_no_weights(edges):
    new_edges = _weight_connections(edges)
    new_edges[new_edges.nonzero()] = 1
    return new_edges


@_dot_edges
def _weight_connections(edges, **_):
    edges = edges.copy()
    edges.data[:] = 1
    return edges


@_dot_edges
def _weight_weighted_connections(edges, **_):
    return edges


@_dot_edges
def _weight_geo(edges, wuu, wvv):
    return _normalize(edges, np.sqrt(wuu()), np.sqrt(wvv()))


@_dot_edges
def _weight_geodeg(edges, **_):
    return _normalize(
        edges,
        np.sqrt(edges.getnnz(axis=0)),
        np.sqrt(edges.getnnz(axis=1)).reshape(edges.shape[1], 1))


@_dot_edges
def _weight_input(edges, wvv, **_):
    return _normalize(edges, wvv())


@_dot_edges
def _weight_output(edges, wuu, **_):
    return _normalize(edges, wuu())


def _norm_min_max(edges, wuu, wvv, f):
    assert isinstance(edges, sp.coo_matrix)
    wuu, wvv = wuu(), wvv()
    edges = edges.copy()
    us, vs = edges.col, edges.row
    edges.data /= f(wuu.A.flatten()[us], wvv.A.flatten()[vs])
    return edges


@_dot_edges
def _weight_min(edges, wuu, wvv):
    return _norm_min_max(edges, wuu, wvv, np.minimum)


@_dot_edges
def _weight_max(edges, wuu, wvv):
    return _norm_min_max(edges, wuu, wvv, np.maximum)


WeightType = namedtuple("WeightType", ["name", "func"])

Weighting = [WeightType(*x) for x in (
    ("No weights", _weight_no_weights),
    ("Number of connections", _weight_connections),
    ("Weighted number of connections", _weight_weighted_connections),
    ("Geometric normalization", _weight_geo),
    ("Geometric normalization by degrees", _weight_geodeg),
    ("Normalization by sum of input weights", _weight_input),
    ("Normalization by sum of output weights", _weight_output),
    ("Normalization by minimal sum of weights", _weight_min),
    ("Normalization by maximal sum of weights", _weight_max)
)]


def _filtered_edges(network, mode_mask, conn_mask, weighted):
    """
    Compute a coo_matrix with network edges between nodes in mode_mask and
    conn_mask
    """
    if not network.number_of_edges():
        return None
    if weighted:
        node1, node2, ws = zip(*network.edges(data="weight"))
        ws = np.array(ws)
    else:
        node1, node2 = zip(*network.edges())
    node1, node2 = np.array(node1), np.array(node2)
    fwd = mode_mask[node1] * conn_mask[node2]
    back = mode_mask[node2] * conn_mask[node1]
    us = np.hstack((node1[fwd], node2[back]))
    vs = np.hstack((node2[fwd], node1[back]))
    if not len(us):
        return sp.coo_matrix((0, 0))
    if weighted:
        data = np.hstack((ws[fwd], ws[back]))
    else:
        data = np.ones(len(us))
    new_indices = np.cumsum(mode_mask) - 1
    return sp.coo_matrix((data, (new_indices[us], vs)))
