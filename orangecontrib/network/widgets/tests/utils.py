import os

import orangecontrib
from Orange.data import Table

from orangecontrib.network.widgets.OWNxFile import OWNxFile
from orangewidget.tests.base import WidgetTest


class NetworkTest(WidgetTest):

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
