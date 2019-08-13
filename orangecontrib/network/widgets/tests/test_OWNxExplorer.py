import os

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from orangewidget.tests.utils import simulate

import orangecontrib
from orangecontrib.network.widgets.OWNxFile import OWNxFile
from orangecontrib.network.widgets.OWNxExplorer import OWNxExplorer


class TestOWNxExplorer(WidgetTest):
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
