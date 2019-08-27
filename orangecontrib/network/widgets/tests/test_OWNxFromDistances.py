import unittest

from Orange.data import Table
from Orange.distance import Euclidean
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxFromDistances import OWNxFromDistances


class TestOWNxFromDistances(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxFromDistances) # type: OWNxFromDistances
        self.data = Table("iris")
        self.distances = Euclidean(self.data)

    def test_minimum_size(self):
        # Disable this test from the base test class
        pass

    def test_node_selection(self):
        self.send_signal(self.widget.Inputs.distances, self.distances)
        net = self.get_output(self.widget.Outputs.network)
        self.assertTrue(net)
        self.assertEqual(net.number_of_nodes(), len(self.data))
        self.widget.controls.node_selection.buttons[1].click()
        net = self.get_output(self.widget.Outputs.network)
        self.assertTrue(net)
        self.assertEqual(net.number_of_nodes(), 84)
        self.widget.controls.node_selection.buttons[2].click()
        net = self.get_output(self.widget.Outputs.network)
        self.assertTrue(net)
        self.assertEqual(net.number_of_nodes(), 36)


if __name__ == "__main__":
    unittest.main()