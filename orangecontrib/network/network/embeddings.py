import numpy as np
import gensim
from gensim.models import Word2Vec
from Orange.data import ContinuousVariable, Table, Domain

from orangecontrib.network.network.base import DirectedEdges


def alias_preprocess(probas_dict):
    """
        Preprocessing step for alias sampling, which enables drawing samples in O(1).
        Returns triple of equal-length lists: [0] contains candidates, [1] contains the probability
        of selecting the candidates after a biased coin flip, [2] contains reference to the alternative choice
        after the flip.

        Arguments
        ---------
        probas_dict: dict
            Keys are nodes/edges, values are their probabilities

        Useful reference: http://www.keithschwarz.com/darts-dice-coins/ (see Vose's Alias Method)
    """
    num_items = len(probas_dict)
    alias, prob = [None] * num_items, [None] * num_items
    # hold unnormalized probabilities < and >= 1, respectively
    small, large = [], []

    items = []
    working_probas = []
    for i, (curr_item, proba) in enumerate(probas_dict.items()):
        items.append(curr_item)
        unnorm_proba = proba * num_items
        working_probas.append(unnorm_proba)
        if unnorm_proba < 1.0:
            small.append((i, unnorm_proba))
        else:
            large.append((i, unnorm_proba))

    while small and large:
        curr_lower, p_lower = small.pop()
        curr_greater, p_greater = large.pop()

        prob[curr_lower] = p_lower
        alias[curr_lower] = curr_greater

        remaining_greater = (p_greater + p_lower) - 1.0
        if remaining_greater < 1.0:
            small.append((curr_greater, remaining_greater))
        else:
            large.append((curr_greater, remaining_greater))

    while large:
        curr_greater, p_greater = large.pop()
        prob[curr_greater] = 1.0

    # can only be non-empty at this point due to numerical instability
    while small:
        curr_lower, p_lower = small.pop()
        prob[curr_lower] = 1.0

    return items, prob, alias


def alias_draw(candidates, prob, alias):
    """ Single draw of alias sampling - first, perform uniform draw between candidates, then do a biased coin flip
    to determine whether to keep the chosen candidate or take its alternative.

    Arguments
    ---------
    candidates: list
        Raw names of candidates
    prob: list
        Probabilities of choosing items in `candidates` during a biased coin flip
    alias: list
        Reference (indices that index `candidates`) to alternative candidates during a biased coin flip
    """
    bucket_id = np.random.randint(0, len(candidates))
    rand_num = np.random.random()

    if rand_num < prob[bucket_id]:
        return candidates[bucket_id]
    else:
        return candidates[alias[bucket_id]]


def normalize(probas, norm_const):
    # either all weights 0 or some positive, some negative - use softmax
    if norm_const == 0.0:
        normed_probas = {}
        for item, proba in probas.items():
            new_proba = np.exp(proba)
            normed_probas[item] = new_proba
            norm_const += new_proba
    else:
        normed_probas = probas

    return {neigh: w / norm_const for neigh, w in normed_probas.items()}


class Node2Vec:
    def __init__(self, p=0.8, q=0.7, walk_len=80, num_walks=10, emb_size=50,
                 window_size=5, num_epochs=1, prefix=None,
                 callbacks=()):
        self.p = p
        self.q = q
        self.walk_len = walk_len
        self.num_walks = num_walks
        self.emb_size = emb_size
        self.window_size = window_size
        self.num_epochs = num_epochs
        self.callbacks = callbacks

        self._node_probas = {}
        self._edge_probas = {}
        if prefix is not None and isinstance(prefix, str):
            self.feature_prefix = prefix
        else:
            self.feature_prefix = "n2v"

    def __call__(self, G):
        if not G.edges:
            raise ValueError("Network has no edges")

        num_nodes = G.number_of_nodes()
        nodes = np.arange(num_nodes)
        # Node->node probas are needed for the initial step, when there's no previous edge to condition on
        self._node_probas = self.setup_nodes(G, nodes)

        edges_coo = G.edges[0].edges.tocoo(copy=False)
        edges = np.column_stack((edges_coo.row, edges_coo.col))
        self._edge_probas = self.setup_edges(G, edges)

        walks = self._simulate_walks(G)
        walks = [list(map(str, walk)) for walk in walks]

        # gensim changed "size" param to "vector_size" in v. 4.0.0
        # https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
        params = dict(window=self.window_size, min_count=0, sg=1, workers=4,
                      callbacks=self.callbacks)
        if gensim.__version__ < "4.0.0":
            params["size"] = self.emb_size
            params["iter"] = self.num_epochs
        else:
            params["vector_size"] = self.emb_size
            params["epochs"] = self.num_epochs
        model = Word2Vec(walks, **params)

        items = G.nodes
        new_attrs = {}
        new_data = np.array([[] for _ in range(num_nodes)])
        class_vars, meta_vars = [], []
        class_data, meta_data = np.array([[] for _ in range(num_nodes)]), np.array([[] for _ in range(num_nodes)])
        if isinstance(items, Table):
            attrs_mask = []
            for attr in items.domain.attributes:
                attrs_mask.append(attr.name not in new_attrs)
                new_attrs[attr.name] = new_attrs.get(attr.name, (len(new_attrs), attr))

            new_data = items.X[:, np.array(attrs_mask, dtype=bool)]
            class_vars, meta_vars = items.domain.class_vars, items.domain.metas
            class_data, meta_data = items.Y, items.metas

        # override existing continuous vars with same names
        for i in range(self.emb_size):
            new_name = "{}_{}".format(self.feature_prefix, i)
            new_attrs[new_name] = (len(new_attrs), ContinuousVariable(new_name))

        new_data = np.hstack((new_data, np.array([model.wv[str(curr_node)] for curr_node in nodes])))
        ordered_attrs = [None] * len(new_attrs)
        for idx, attr in new_attrs.values():
            ordered_attrs[idx] = attr
        new_domain = Domain(ordered_attrs, class_vars, meta_vars)
        new_items = Table(new_domain, new_data, class_data, meta_data)
        return new_items

    def _single_walk(self, start_node):
        # assumes node and edge probabilities have been computed
        curr_walk = [start_node]

        nodes, prob, alias = self._node_probas[start_node]
        # isolated starting node
        if not nodes:
            return curr_walk
        # initial step (no previous node to condition on)
        prev_node, curr_node = start_node, alias_draw(nodes, prob, alias)
        curr_walk.append(curr_node)

        for _ in range(self.walk_len - 2):
            edges, prob, alias = self._edge_probas[(prev_node, curr_node)]
            if len(edges) == 0:
                break

            prev_node, curr_node = alias_draw(edges, prob, alias)
            curr_walk.append(curr_node)

        return curr_walk

    def _simulate_walks(self, G):
        # assumes node and edge probabilities have been computed
        walks = []
        nodes = np.arange(G.number_of_nodes())

        for idx_walk in range(self.num_walks):
            np.random.shuffle(nodes)
            for curr_node in nodes:
                walks.append(self._single_walk(start_node=curr_node))

        return walks

    def setup_nodes(self, G, nodes):
        return {n: alias_preprocess(self.node_probas(G, n)) for n in nodes}

    def node_probas(self, G, from_node):
        probas = {}
        norm_const = 0.0
        edges = G.edges[0]
        edges = edges.edges if isinstance(edges, DirectedEdges) else edges.twoway_edges

        for neigh_node in G.outgoing(from_node):
            edge_w = edges[from_node, neigh_node]
            norm_const += edge_w
            probas[neigh_node] = edge_w

        probas = normalize(probas, norm_const)
        return probas

    def setup_edges(self, G, edges):
        probas = {}
        if isinstance(G.edges[0], DirectedEdges):
            for u, v in edges:
                probas[(u, v)] = alias_preprocess(self.edge_probas(G, u, v))
        else:
            for u, v in edges:
                probas[(u, v)] = alias_preprocess(self.edge_probas(G, u, v))
                probas[(v, u)] = alias_preprocess(self.edge_probas(G, v, u))
        return probas

    def edge_probas(self, G, u, v):
        """ Compute transition probas from `v` to neighbors, taking into account that edge u->v was just traversed. """
        probas = {}
        norm_const = 0.0
        edges = G.edges[0]
        edges = edges.edges if isinstance(edges, DirectedEdges) else edges.twoway_edges

        # for checking if edge exists
        edges_coo = edges.tocoo(copy=False)
        edges_coo = set(zip(edges_coo.row, edges_coo.col))

        for neigh_of_dst in G.outgoing(v):
            edge_w = edges[v, neigh_of_dst]
            if neigh_of_dst == u:
                edge_w = edge_w / self.p
            elif (u, neigh_of_dst) in edges_coo:
                edge_w = edge_w
            else:
                edge_w = edge_w / self.q

            probas[(v, neigh_of_dst)] = edge_w
            norm_const += edge_w

        probas = normalize(probas, norm_const)
        return probas
