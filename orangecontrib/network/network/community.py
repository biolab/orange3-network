"""
.. index:: Community Detection in Graphs

.. index::
   single: Network; Community Detection in Graphs

*****************************
Community Detection in Graphs
*****************************

"""

import random
import itertools
import numpy as np

from Orange.data import Domain, Table, DiscreteVariable


def add_results_to_items(G, labels, var_name):
    items = G.nodes

    new_attr = DiscreteVariable(
        var_name, values=["C%d" % (x + 1) for x in set(labels.values())])
    new_data = np.array(list(labels.values())).reshape(-1, 1)

    if items is not None:
        # a unique `var_name` is assumed to be obtained in OWNxClustering prior
        # to calling this; only check metas to enable overriding results on
        # multiple reruns
        effective_metas = [idx for idx, meta in enumerate(items.domain.metas)
                           if meta.name != var_name]
        metas = np.take(items.domain.metas, effective_metas).tolist()
        metas.append(new_attr)

        meta_data = items.metas[:, effective_metas]
        meta_data = np.hstack((meta_data, new_data))

        domain = Domain(items.domain.attributes, items.domain.class_vars, metas)
        data = Table(domain, items.X, items.Y, meta_data, items.W)
        G.nodes = data
    else:
        empty = np.array([[] for _ in range(new_data.shape[0])])
        domain = Domain([], [], metas=[new_attr])
        data = Table(domain, empty, empty, new_data)
        G.node = data


class CommunityDetection(object):
    def __init__(self, algorithm, **kwargs):
        self.algorithm = algorithm
        self.kwargs = kwargs

    def __call__(self, G):
        return self.algorithm(G, **self.kwargs)


def label_propagation_hop_attenuation(G, iterations=1000,
                                      delta=0.1, node_degree_preference=0, seed=None):
    """Label propagation for community detection, Leung et al., 2009

    :param G: A Orange graph.
    :type G: Orange.network.Graph

    :param iterations: The max. number of iterations if no convergence.
    :type iterations: int

    :param delta: The hop attenuation factor.
    :type delta: float

    :param node_degree_preference: The power on node degree factor.
    :type node_degree_preference: float

    :param seed: Seed for replicable clustering.
    :type seed: int
    """

    if G.edges[0].directed:
        raise ValueError("Undirected graph expected")

    if seed is not None:
        random.seed(seed)

    vertices = list(range(G.number_of_nodes()))
    degrees = dict(enumerate(G.degrees()))
    labels = dict(zip(vertices, range(G.number_of_nodes())))
    scores = dict(zip(vertices, [1] * G.number_of_nodes()))
    m = node_degree_preference

    for i in range(iterations):
        random.shuffle(vertices)
        stop = 1
        for v in vertices:
            neighbors = list(G.neighbours(v))
            if len(neighbors) == 0:
                continue

            lbls = sorted(((G.edges[0].edges[v, u], labels[u], u)
                           for u in neighbors), key=lambda x: x[1])
            lbls = [(sum(scores[u] * degrees[u] ** m * weight for weight,
                         _u_label, u in group), label)
                    for label, group in itertools.groupby(lbls, lambda x: x[1])]
            max_score = max(lbls)[0]
            max_lbls = [label for score, label in lbls if score >= max_score]

            # only change label if it is not already one of the
            # preferred (maximal) labels
            if labels[v] not in max_lbls:
                labels[v] = random.choice(max_lbls)
                scores[v] = max(0, max(scores[u] for u in neighbors \
                                       if labels[u] == labels[v]) - delta)
                stop = 0

        # if stopping condition is satisfied (no label switched color)
        if stop:
            break

    labels = remap_labels(labels)

    return labels


def label_propagation(G, iterations=1000, seed=None):
    """Label propagation for community detection, Raghavan et al., 2007

    :param G: A Orange graph.
    :type G: Orange.network.Graph

    :param iterations: The maximum number of iterations if there is no convergence.
    :type iterations: int

    :param seed: Seed for replicable clustering.
    :type seed: int
    """
    if seed is not None:
        random.seed(seed)

    vertices = list(range(G.number_of_nodes()))
    labels = dict(zip(vertices, range(G.number_of_nodes())))

    def next_label(neighbors):
        """Updating rule as described by Raghavan et al., 2007

        Return a list of possible node labels with equal probability.
        """
        lbls = sorted(labels[u] for u in neighbors)
        lbls = [(len(list(c)), l) for l, c in itertools.groupby(lbls)]
        m = max(lbls)[0]
        return [l for c, l in lbls if c >= m]

    for i in range(iterations):
        random.shuffle(vertices)
        stop = 1
        for v in vertices:
            nbh = list(G.neighbours(v))
            if len(nbh) == 0:
                continue

            max_lbls = next_label(nbh)

            if labels[v] not in max_lbls:
                stop = 0

            labels[v] = random.choice(max_lbls)

        # if stopping condition might be satisfied, check it
        # stop when no label would switch anymore
        if stop:
            for v in vertices:
                nbh = list(G.neighbours(v))
                if len(nbh) == 0:
                    continue
                max_lbls = next_label(nbh)
                if labels[v] not in max_lbls:
                    stop = 0
                    break

            if stop:
                break

    labels = remap_labels(labels)

    return labels


def remap_labels(labels):
    unique_labels = sorted(set(labels.values()))
    dict_labels = {label:i for i, label in enumerate(unique_labels)}
    return {key: dict_labels[val] for key, val in labels.items()}
