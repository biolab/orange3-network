import os
import numpy as np
import scipy.sparse as sp

import orangecontrib
from Orange.data import Table

from orangecontrib.network import Network
from orangecontrib.network.network.base import DirectedEdges, UndirectedEdges
from orangecontrib.network.widgets.OWNxFile import OWNxFile
from orangewidget.tests.base import WidgetTest


def _create_net(edges, n=None, directed=False):
    edge_cons = DirectedEdges if directed else UndirectedEdges
    row, col, data = zip(*edges)
    if n is None:
        n = max(*row, *col) + 1
    return Network(np.arange(n), edge_cons(sp.coo_matrix((data, (row, col)), shape=(n, n))))


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
