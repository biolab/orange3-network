import unittest

import numpy as np

from Orange.data import Table
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxFromDistances import OWNxFromDistances


class TestOWNxFromDistances(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxFromDistances) # type: OWNxFromDistances
        self.data = Table("iris")
        self.distances = Euclidean(self.data)

        # When converted to a graph, this has the following components:
        # At threshold 0.5:  {1, 6} and disconnected {0}, {2}, {3}, {4}, {5}
        # At threshold 1 {0, 1, 2, 6}, {3, 5}, {4}
        # At threshold 2 {0, 1, 2, 3, 5, 6}, {4}
        m = np.full((7, 7), 10.0)
        m[1, 6] = m[6, 1] = 0.5

        m[0, 1] = m[1, 2] = m[2, 6] = m[0, 6] = 1
        m[1, 0] = m[2, 1] = m[6, 2] = m[6, 0] = 1

        m[3, 5] = m[5, 3] = 1

        m[2, 3] = m[3, 2] = 2
        self.distances1 = DistMatrix(m)

    def test_minimum_size(self):
        # Disable this test from the base test class
        pass

    def _set_threshold(self, value):
        self.widget.spinUpperThreshold = value
        mat = self.widget.matrix_values
        self.widget.percentil = 100 * np.searchsorted(mat, value) / len(mat)

    def test_node_selection(self):
        self.send_signal(self.widget.Inputs.distances, self.distances1)
        self.widget.excludeLimit = 2
        self._set_threshold(1.5)

        self.widget.controls.node_selection.buttons[0].click()
        net = self.get_output(self.widget.Outputs.network)
        self.assertTrue(net)
        self.assertEqual(net.number_of_nodes(), 7)

        self.widget.controls.node_selection.buttons[1].click()
        net = self.get_output(self.widget.Outputs.network)
        self.assertTrue(net)
        self.assertEqual(net.number_of_nodes(), 6)
        self.widget.controls.node_selection.buttons[2].click()
        net = self.get_output(self.widget.Outputs.network)
        self.assertTrue(net)
        self.assertEqual(net.number_of_nodes(), 4)

    def test_no_crash_on_zero_distance(self):
        """ Test that minimum distance 0 does not make the widget automatically set the distance threshold under 0,
        causing no nodes to satisfy condition"""
        dist = Euclidean(self.data, axis=1)

        self.widget.percentil = 100.0
        self.send_signal(self.widget.Inputs.distances, dist)
        net = self.get_output(self.widget.Outputs.network)
        self.assertTrue(net)
        self.assertEqual(net.number_of_nodes(), len(self.data))

    def test_no_crash_on_single_instance(self):
        """Test that single instance does not crash widget due to distance matrix having no valid distances"""
        dist = Euclidean(self.data[:1], axis=1)

        self.send_signal(self.widget.Inputs.distances, dist)
        net = self.get_output(self.widget.Outputs.network)
        self.assertTrue(net)
        self.assertEqual(net.number_of_nodes(), 1)


if __name__ == "__main__":
    unittest.main()
