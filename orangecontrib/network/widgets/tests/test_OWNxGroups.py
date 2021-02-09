import unittest
from unittest.mock import Mock
from math import sqrt

import numpy as np
from scipy import sparse as sp

from Orange.data import Table, Domain, DiscreteVariable
from Orange.widgets.tests.base import simulate

from orangecontrib.network import Network
from orangecontrib.network.widgets.OWNxGroups import OWNxGroups
from orangecontrib.network.widgets.tests.utils import NetworkTest


class TestOWNxGroups(NetworkTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxGroups)

    def test_inputs(self):
        Outputs = self.widget.Outputs
        network = self._read_network()
        table = self._read_items()

        # send network with items
        self.send_signal(self.widget.Inputs.network, network)
        names = [a.name for a in self.widget.effective_data.domain.attributes]
        self.assertListEqual(["department"], names)
        self.assertIsInstance(self.get_output(Outputs.network), Network)

        # send subset of items as data
        self.send_signal(self.widget.Inputs.data, table[:-1])
        self.assertTrue(self.widget.Error.data_size_mismatch.is_shown())
        self.assertIsNone(self.widget.effective_data)
        self.assertIsNone(self.get_output(self.widget.Outputs.network))

        # remove sent items
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.data_size_mismatch.is_shown())
        names = [a.name for a in self.widget.effective_data.domain.attributes]
        self.assertListEqual(["department"], names)
        self.assertIsInstance(self.get_output(Outputs.network), Network)

        # send all items (with slightly different domain) as data
        self.send_signal(self.widget.Inputs.data, table)
        names = [a.name for a in self.widget.effective_data.domain.attributes]
        self.assertListEqual(["department"], names)
        self.assertIsInstance(self.get_output(Outputs.network), Network)

        # remove network
        self.send_signal(self.widget.Inputs.network, None)
        self.assertTrue(self.widget.Warning.no_graph_found.is_shown())
        self.assertIsNone(self.widget.effective_data)
        self.assertIsNone(self.get_output(self.widget.Outputs.network))

        # remove items
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.no_graph_found.is_shown())

    def test_outputs(self):
        in_net = self._read_network()
        self.send_signal(self.widget.Inputs.network, in_net)
        network = self.get_output(self.widget.Outputs.network)

        self.assertIsInstance(network, Network)
        self.assertIsInstance(network.nodes, Table)
        np.testing.assert_equal(set(network.nodes.X.flatten()), set(range(3)))
        self.assertEqual(
            set(zip(*network.edges[0].edges.nonzero())),
            {(0, 1), (1, 2), (0, 2)})

    def test_no_discrete_features(self):
        Outputs = self.widget.Outputs
        network = self._read_network("airtraffic.net")
        self.send_signal(self.widget.Inputs.network, network)
        self.assertIsInstance(self.get_output(Outputs.network), Network)

        network = self._read_network("mips_c2_cp_leu.net")
        self.send_signal(self.widget.Inputs.network, network)
        self.assertTrue(self.widget.Warning.no_discrete_features.is_shown())
        self.assertIsNone(self.get_output(Outputs.network))

        self.send_signal(self.widget.Inputs.network, None)
        self.assertFalse(self.widget.Warning.no_discrete_features.is_shown())

    def test_missing_values(self):
        network = self._read_network("airtraffic.net")
        self.send_signal(self.widget.Inputs.network, network)
        fname = network.nodes.domain.metas[-1].name
        simulate.combobox_activate_item(self.widget.controls.feature, fname)
        self.assertIsInstance(
            self.get_output(self.widget.Outputs.network), Network)

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.network, self._read_network())
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.network, None)
        self.widget.send_report()

    def test_disable_normalize(self):
        widget = self.widget
        buttons = widget.controls.weighting.buttons
        normalize = widget.controls.normalize

        buttons[widget.NoWeights].click()
        self.assertFalse(normalize.isEnabled())
        buttons[widget.WeightByDegrees].click()
        self.assertFalse(normalize.isEnabled())
        buttons[widget.WeightByWeights].click()
        self.assertTrue(normalize.isEnabled())
        buttons[widget.WeightByDegrees].click()
        self.assertFalse(normalize.isEnabled())

    def test_commit_on_normalization_change(self):
        widget = self.widget
        check = widget.controls.normalize
        widget.commit = Mock()

        check.click()
        widget.commit.assert_called()
        widget.commit.reset_mock()

        check.click()
        widget.commit.assert_called()

    def test_weights(self):
        label_var = DiscreteVariable("label", values=tuple("abcde"))
        domain = Domain([label_var], [])
        items = Table.from_numpy(domain, np.array([[0, 0, 1, 3, 4, 0, 1, 2, 4]]).T)
        src, dst, weights = np.array(
            [[0, 1, 5],
             [1, 2, 4],
             [3, 4, 6],
             [0, 5, 3],
             [1, 6, 1],
             [2, 7, 2],
             [3, 8, 8],
             [4, 8, 7],
             [5, 6, 2]]).T
        edges = sp.coo_matrix((weights.astype(float), (src, dst)), shape=(9, 9))
        network = Network(items, edges)
        expected = np.zeros((5, 5), dtype=float)

        widget = self.widget
        buttons = widget.controls.weighting.buttons
        self.send_signal(widget.Inputs.network, network)

        buttons[widget.NoWeights].click()
        groups = widget._map_network()
        for i, j in [(0, 1), (1, 2), (3, 4)]:
            expected[i, j] = 1
        np.testing.assert_equal(groups.edges[0].edges.todense(), expected)

        buttons[widget.WeightByDegrees].click()
        groups = widget._map_network()
        expected[0, 1] = 3
        expected[1, 2] = 1
        expected[3, 4] = 2
        np.testing.assert_equal(groups.edges[0].edges.todense(), expected)

        buttons[widget.WeightByWeights].click()
        widget.normalize = False
        groups = widget._map_network()
        expected[0, 1] = 7
        expected[1, 2] = 2
        expected[3, 4] = 14
        np.testing.assert_equal(groups.edges[0].edges.todense(), expected)

        widget.normalize = True
        groups = widget._map_network()
        expected[0, 1] = 1 / sqrt(10 * 3) + 4 / sqrt(10 * 6) + 2 / sqrt(5 * 3)
        expected[1, 2] = 2 / sqrt(6 * 2)
        expected[3, 4] = 6 / sqrt(14 * 13) + 8 / sqrt(14 * 15)
        np.testing.assert_equal(groups.edges[0].edges.todense(), expected)


if __name__ == "__main__":
    unittest.main()
