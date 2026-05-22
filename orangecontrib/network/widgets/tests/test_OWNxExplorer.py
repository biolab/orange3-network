import unittest
from unittest.mock import Mock

import numpy as np
from AnyQt.QtTest import QSignalSpy

from orangewidget.tests.utils import simulate

from orangecontrib.network.widgets.tests.utils import NetworkTest
from orangecontrib.network import Network
from orangecontrib.network.widgets.OWNxExplorer import OWNxExplorer, \
    WEIGHTS_COMBO_ITEM


class TestOWNxExplorer(NetworkTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxExplorer)  # type: OWNxExplorer
        self.network = self._read_network("lastfm.net")
        self.data = self._read_items("lastfm.tab")

        self.davis_net = self._read_network("davis.net")
        self.davis_data = self._read_items("davis.tsv")

        self.railway = self._read_network("railway.net")

    def test_minimum_size(self):
        # Disable this test from the base test class
        pass


class TestOWNxExplorerWithLayout(TestOWNxExplorer):
    def test_empty_network(self):
        net = Network([], [])
        # should not crash
        self.send_signal(self.widget.Inputs.network, net)

    def test_positions_hint(self):
        self.send_signal(self.widget.Inputs.network, self.davis_net)
        self.wait_until_finished()

        # Store positions
        positions = self.widget.positions
        self.assertIsNotNone(positions)
        # Intermediate test: positions hint should be available after optimization finishes
        self.assertIsNotNone(self.widget.positions_hint)

        # Clear the network and check that positions are cleared as well
        self.send_signal(self.widget.Inputs.network, None)
        self.assertIsNone(self.widget.positions)

        # Send the same network again and check that positions are the same as
        # before. If hint was not used, positions would be different
        self.send_signal(self.widget.Inputs.network, self.davis_net)
        np.testing.assert_equal(self.widget.positions, positions)

        # Send a different network and check that positions are different
        # This must, basically, not crash
        self.send_signal(self.widget.Inputs.network, self.railway)
        self.wait_until_finished()

        self.assertNotEqual(self.widget.positions.tolist(), positions.tolist())
        positions = self.widget.positions

        # Now change the edges and see that positions are updated
        edges = self.railway.edges[0].edges.copy()
        edges.indices[0] += 1
        new_network = Network(self.railway.nodes, type(self.railway.edges[0])(edges))
        self.send_signal(self.widget.Inputs.network, new_network)
        self.wait_until_finished()

        self.assertNotEqual(self.widget.positions.tolist(), positions.tolist())


class TestOWNxEplorerWithoutLayout(TestOWNxExplorer):
    def setUp(self):
        super().setUp()
        self.widget.relayout = Mock()

    def test_too_many_labels(self):
        # check that Warning is shown when attribute
        # with too many labels is chosen
        # GH - 120
        self.send_signal(self.widget.Inputs.network, self.network)
        self.send_signal(self.widget.Inputs.node_data, self.data)
        simulate.combobox_activate_item(self.widget.controls.attr_label,
                                        self.data.domain.metas[0].name)
        self.assertTrue(self.widget.Warning.too_many_labels.is_shown())
        simulate.combobox_activate_index(self.widget.controls.attr_label, 0)
        self.assertFalse(self.widget.Warning.too_many_labels.is_shown())

    def test_subset_selection(self):
        # test if selecting from the graph works
        self.send_signal(self.widget.Inputs.network, self.network)
        self.assertEqual(self.widget.nSelected, 0)
        self.send_signal(self.widget.Inputs.node_data, self.data)
        self.widget.graph.selection_select(np.arange(0, 5))
        outputs = self.widget.Outputs
        self.assertIsInstance(self.get_output(outputs.subgraph), Network)
        self.assertEqual(self.widget.nSelected, 5)

    def test_get_reachable(self):
        # gene label indices, which are equal to their numbers assigned in the .net file - 1
        GENES = {"IDS": 70, "GNS": 36, "BLVRB": 61, "HMOX1": 5, "BLVRA": 62,
                 "PSMA2": 6, "PSMA4": 8, "PSMA5": 9, "PSMA6": 7}
        self.send_signal(self.widget.Inputs.network, self._read_network("leu_by_genesets.net"))

        self.assertSetEqual(set(self.widget.get_reachable([GENES["PSMA2"]])),
                            {GENES["PSMA2"], GENES["PSMA4"], GENES["PSMA5"], GENES["PSMA6"]})
        # test that zero-weight edges do not get dropped due to sparse matrix representation
        self.assertSetEqual(set(self.widget.get_reachable([GENES["IDS"]])),
                            {GENES["IDS"], GENES["GNS"]})
        self.assertSetEqual(set(self.widget.get_reachable([GENES["BLVRB"]])),
                            {GENES["BLVRB"], GENES["HMOX1"], GENES["BLVRA"]})

    def test_edge_weights(self):
        self.send_signal(self.widget.Inputs.network, self.davis_net)
        self.widget.edge_label_variable = WEIGHTS_COMBO_ITEM
        self.widget.graph.label_only_selected = True

        # Mark nodes with many connections (multiple): should show the weights for edges between marked nodes only
        self.widget.mark_min_conn = 8
        self.widget.set_mark_mode(9)
        self.assertEqual(len(self.widget.graph.edge_labels), 12)

        # Reset to default (no selection) and check that the labels disappear
        self.widget.set_mark_mode(0)
        self.assertEqual(len(self.widget.graph.edge_labels), 0)

        # Mark nodes with most connections (single): should show all its edges' weights
        self.widget.set_mark_mode(10)
        self.assertEqual(len(self.widget.graph.edge_labels), 14)

    def test_input_subset(self):
        self.send_signal(self.widget.Inputs.network, self.davis_net)
        self.send_signal(self.widget.Inputs.node_data, self.davis_data)
        sub_mask = self.widget.get_subset_mask()
        self.assertIsNone(sub_mask)

        self.send_signal(self.widget.Inputs.node_subset, self.davis_data[:3])
        sub_mask = self.widget.get_subset_mask()
        num_subset_nodes = np.sum(sub_mask)
        self.assertEqual(num_subset_nodes, 3)

        self.send_signal(self.widget.Inputs.node_subset, None)
        sub_mask = self.widget.get_subset_mask()
        self.assertIsNone(sub_mask)

    def test_report(self):
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.network, self.davis_net)
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.node_data, self.davis_data)
        self.widget.send_report()

    def test_clear_selection_on_no_data(self):
        self.widget.relayout = Mock()
        self.send_signal(self.widget.Inputs.network, self.network)
        self.widget.graph.selection_select(np.arange(0, 5))
        self.assertEqual(self.widget.nSelected, 5)
        self.assertIsNotNone(self.widget.selection)
        self.send_signal(self.widget.Inputs.network, None)
        self.assertEqual(self.widget.nSelected, 0)
        self.assertIsNone(self.widget.selection)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_node_settings(self):
        self.send_signal(self.widget.Inputs.network, self.davis_net)
        self.send_signal(self.widget.Inputs.node_data, self.davis_data)

        self.widget.attr_color = self.davis_data.domain["id"]
        self.widget.attr_shape = self.davis_data.domain["role"]
        self.widget.attr_label = self.davis_data.domain["name"]
        self.widget.attr_size = self.davis_data.domain["id"]
        self.send_signal(self.widget.Inputs.node_data, None)
        self.assertIsNone(self.widget.attr_color)
        self.assertIsNone(self.widget.attr_shape)
        self.assertIsNone(self.widget.attr_label)
        self.assertIsNone(self.widget.attr_size)

        self.send_signal(self.widget.Inputs.network, None)
        self.assertIsNone(self.widget.attr_color)
        self.assertIsNone(self.widget.attr_shape)
        self.assertIsNone(self.widget.attr_label)
        self.assertIsNone(self.widget.attr_size)

if __name__ == "__main__":
    unittest.main()
