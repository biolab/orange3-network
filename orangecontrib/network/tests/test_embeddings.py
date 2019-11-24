import unittest

import numpy as np
import scipy.sparse as sp
from Orange.data import Table, ContinuousVariable, Domain

from orangecontrib.network import Network
from orangecontrib.network.network.base import DirectedEdges, UndirectedEdges
from orangecontrib.network.network.embeddings import Node2Vec


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        row, col, w = zip(*((1, 2, 1.0), (1, 3, 3.0), (2, 3, 1.0), (2, 6, 0.5), (3, 4, 1.0), (4, 5, 1.0), (4, 7, -1.0),
                            (5, 6, 0.0), (6, 5, 0.1), (6, 2, 0.1)))
        dir_edges = DirectedEdges(sp.csr_matrix((w, (row, col)), shape=(8, 8)))
        self.toy_directed = Network(np.arange(8), dir_edges)

        row, col, w = zip(*((1, 2, 1.0), (1, 3, 3.0), (2, 3, 1.0), (2, 6, 0.5), (3, 4, 1.0), (4, 5, 1.0), (4, 7, -1.0),
                            (5, 6, 0.1)))
        undir_edges = UndirectedEdges(sp.csr_matrix((w, (row, col)), shape=(8, 8)))
        self.toy_undirected = Network(np.arange(8), undir_edges)

    def test_node_probas(self):
        """ Test that node probabilities get calculated correctly """
        n2v = Node2Vec()
        # nowhere to go from isolated node
        self.assertEqual(len(n2v.node_probas(self.toy_directed, 0)), 0)
        # should not have division by zero when weights of edges to neighbors sum to 0
        self.assertDictEqual(n2v.node_probas(self.toy_directed, 5), {6: 1.0})
        probas = n2v.node_probas(self.toy_directed, 4)
        self.assertAlmostEqual(probas[5], 0.881, places=3)
        self.assertAlmostEqual(probas[7], 0.119, places=3)

        self.assertDictEqual(n2v.node_probas(self.toy_directed, 1), {2: 0.25, 3: 0.75})
        self.assertDictEqual(n2v.node_probas(self.toy_undirected, 3), {1: 0.6, 2: 0.2, 4: 0.2})

    def test_edge_probas(self):
        """ Test that edge probabilities get calculated appropriately based on shortest distance between previous node
            and next node (equations in 'Search bias' section of node2vec paper) """
        n2v = Node2Vec(p=0.8, q=0.5)
        edge_probas = n2v.edge_probas(self.toy_directed, 3, 4)
        self.assertAlmostEqual(edge_probas[(4, 5)], 0.982, places=3)  # d_tx = 2
        self.assertAlmostEqual(edge_probas[(4, 7)], 0.018, places=3)  # d_tx = 2
        edge_probas = n2v.edge_probas(self.toy_directed, 1, 2)
        self.assertAlmostEqual(edge_probas[(2, 3)], 0.5)  # d_tx = 1
        edge_probas = n2v.edge_probas(self.toy_directed, 5, 6)
        self.assertAlmostEqual(edge_probas[(6, 5)], 0.385, places=3)  # d_tx = 0

        edge_probas = n2v.edge_probas(self.toy_undirected, 1, 2)
        self.assertAlmostEqual(edge_probas[(2, 1)], 0.385, places=3)  # d_tx = 0
        self.assertAlmostEqual(edge_probas[(2, 3)], 0.308, places=3)  # d_tx = 1
        self.assertAlmostEqual(edge_probas[(2, 6)], 0.308, places=3)  # d_tx = 2

    def test_call(self):
        n2v = Node2Vec(num_walks=10, walk_len=80, emb_size=300)
        embeddings = n2v(self.toy_directed)
        self.assertEqual(embeddings.X.shape, (self.toy_directed.number_of_nodes(), 300))

        # check that domain is extended and that the  existing attributes do not change places
        empty = np.array([[] for _ in range(8)])
        data = Table(Domain([ContinuousVariable("var1")]), np.array([[i] for i in range(8)]), empty, empty)
        toy_net_with_data = Network(data, self.toy_directed.edges[0])
        extended_data = n2v(toy_net_with_data)

        self.assertEqual(extended_data.X.shape, (toy_net_with_data.number_of_nodes(), 1 + 300))
        np.testing.assert_array_almost_equal(extended_data.X[:, 0], np.arange(8))


if __name__ == '__main__':
    unittest.main()
