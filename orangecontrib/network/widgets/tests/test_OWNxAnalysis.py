import unittest
import scipy.sparse as sp

from orangecontrib.network.network.base import Network, UndirectedEdges
from orangecontrib.network.widgets.OWNxAnalysis import density, avg_degree
from orangecontrib.network.widgets.tests.utils import NetworkTest, _create_net


class TestOWNxAnalysis(NetworkTest):
    def setUp(self):
        self.small_undir = _create_net(((0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)), n=5)
        self.small_dir = _create_net(((0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)), n=5, directed=True)
        self.empty_net = Network([], UndirectedEdges(sp.coo_matrix((0, 0))))

    def test_density(self):
        self.assertAlmostEqual(density(self.small_dir), 0.2)
        self.assertAlmostEqual(density(self.small_undir), 0.4)
        # n = 0 and n = 1 should not cause division problems
        self.assertAlmostEqual(density(self.empty_net), 0.0)
        self.assertAlmostEqual(density(self.empty_net), 0.0)

    def test_average_degree(self):
        self.assertAlmostEqual(avg_degree(self.small_dir), 0.8)
        self.assertAlmostEqual(avg_degree(self.small_undir), 1.6)
        self.assertAlmostEqual(avg_degree(self.empty_net), 0.0)


if __name__ == '__main__':
    unittest.main()
