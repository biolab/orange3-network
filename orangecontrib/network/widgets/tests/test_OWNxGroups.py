import os
import unittest

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, simulate
import orangecontrib.network
from orangecontrib.network import Graph
from orangecontrib.network.widgets.OWNxGroups import OWNxGroups
from orangecontrib.network.widgets.OWNxFile import OWNxFile


class TestOWNxGroups(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxGroups)

    def test_inputs(self):
        Outputs = self.widget.Outputs
        network = self._read_network()
        table = self._read_items()

        # send network with items
        self.send_signal(self.widget.Inputs.network, network)
        names = [a.name for a in self.widget.effective_data.domain.attributes]
        self.assertListEqual(["department", "x", "y"], names)
        self.assertIsInstance(self.get_output(Outputs.network), Graph)

        # send subset of items as data
        self.send_signal(self.widget.Inputs.data, table[:-1])
        self.assertTrue(self.widget.Error.data_size_mismatch.is_shown())
        self.assertIsNone(self.widget.effective_data)
        self.assertIsNone(self.get_output(self.widget.Outputs.network))

        # remove sent items
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.data_size_mismatch.is_shown())
        names = [a.name for a in self.widget.effective_data.domain.attributes]
        self.assertListEqual(["department", "x", "y"], names)
        self.assertIsInstance(self.get_output(Outputs.network), Graph)

        # send all items (with slightly different domain) as data
        self.send_signal(self.widget.Inputs.data, table)
        names = [a.name for a in self.widget.effective_data.domain.attributes]
        self.assertListEqual(["department"], names)
        self.assertIsInstance(self.get_output(Outputs.network), Graph)

        # send network without items
        network.set_items(None)
        self.send_signal(self.widget.Inputs.network, network)
        names = [a.name for a in self.widget.effective_data.domain.attributes]
        self.assertListEqual(["department"], names)
        self.assertIsInstance(self.get_output(Outputs.network), Graph)

        # remove network
        self.send_signal(self.widget.Inputs.network, None)
        self.assertTrue(self.widget.Warning.no_graph_found.is_shown())
        self.assertIsNone(self.widget.effective_data)
        self.assertIsNone(self.get_output(self.widget.Outputs.network))

        # remove items
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.no_graph_found.is_shown())

    def test_outputs(self):
        self.send_signal(self.widget.Inputs.network, self._read_network())
        network = self.get_output(self.widget.Outputs.network)

        self.assertIsInstance(network, Graph)
        self.assertSetEqual(set(network.nodes()), set(range(3)))
        edges_ = list(network.edges())
        for n1, n2 in [(0, 1), (1, 2), (0, 2)]:
            self.assertTrue((n1, n2) in edges_)

        self.assertIsInstance(network.items(), Table)

    def test_no_discrete_features(self):
        Outputs = self.widget.Outputs
        network = self._read_network("airtraffic.net")
        self.send_signal(self.widget.Inputs.network, network)
        self.assertIsInstance(self.get_output(Outputs.network), Graph)

        network = self._read_network("mips_c2_cp_leu.net")
        self.send_signal(self.widget.Inputs.network, network)
        self.assertTrue(self.widget.Warning.no_discrete_features.is_shown())
        self.assertIsNone(self.get_output(Outputs.network))

        self.send_signal(self.widget.Inputs.network, None)
        self.assertFalse(self.widget.Warning.no_discrete_features.is_shown())

    def test_missing_values(self):
        network = self._read_network("airtraffic.net")
        self.send_signal(self.widget.Inputs.network, network)
        fname = network.items().domain.metas[-2].name
        simulate.combobox_activate_item(self.widget.controls.feature, fname)
        self.assertIsInstance(
            self.get_output(self.widget.Outputs.network), Graph)

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.network, self._read_network())
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.network, None)
        self.widget.report_button.click()

    def _read_network(self, filename=None):
        owfile = self.create_widget(OWNxFile)
        owfile.open_net_file(self._get_filename(filename, "n"))
        return self.get_output(owfile.Outputs.network, widget=owfile)

    def _read_items(self, filename=None):
        return Table(self._get_filename(filename))

    def _get_filename(self, filename, mode="d"):
        path = os.path.split(orangecontrib.network.__file__)[0]
        if filename is None:
            path = os.path.join(path, "widgets", "tests")
            filename = "test_items.tab" if mode == "d" else "test.net"
        return os.path.join(path, os.path.join("networks", filename))


if __name__ == "__main__":
    unittest.main()
