import numpy as np

from orangecontrib.network.widgets.tests.utils import NetworkTest
from orangewidget.tests.utils import simulate

from orangecontrib.network import Network
from orangecontrib.network.widgets.OWNxExplorer import OWNxExplorer


class TestOWNxExplorer(NetworkTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxExplorer)  # type: OWNxExplorer
        self.network = self._read_network("lastfm.net")
        self.data = self._read_items("lastfm.tab")

    def test_minimum_size(self):
        # Disable this test from the base test class
        pass

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
        self.send_signal(self.widget.Inputs.node_data, self.data)
        self.widget.graph.selection_select(np.arange(0, 5))
        outputs = self.widget.Outputs
        self.assertIsInstance(self.get_output(outputs.subgraph), Network)
