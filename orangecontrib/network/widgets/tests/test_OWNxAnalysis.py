import unittest
import scipy.sparse as sp

from orangecontrib.network.network.base import Network, UndirectedEdges
from orangecontrib.network.widgets.OWNxAnalysis import density, avg_degree, OWNxAnalysis
from orangecontrib.network.widgets.tests.utils import NetworkTest, _create_net


class TestOWNxAnalysis(NetworkTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxAnalysis)
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

    def test_show_valid_indices_by_graph_type(self):
        """ Test that some statistics get disabled based on type of network on input """
        self.send_signal(self.widget.Inputs.network, self.small_undir)
        self.assertFalse(self.widget.method_cbs["number_strongly_connected_components"].isEnabled())
        self.assertFalse(self.widget.method_cbs["number_weakly_connected_components"].isEnabled())

        self.send_signal(self.widget.Inputs.network, self.small_dir)
        self.assertTrue(self.widget.method_cbs["number_strongly_connected_components"].isEnabled())
        self.assertTrue(self.widget.method_cbs["number_weakly_connected_components"].isEnabled())


if __name__ == '__main__':
    unittest.main()
