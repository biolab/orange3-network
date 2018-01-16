import networkx as nx

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxExplorer import OWNxExplorer
from orangecontrib.network import readwrite


class TestOWNxExplorer(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWNxExplorer)  # type: OWNxExplorer

    def _create_graph(self):
        func = lambda n: nx.barbell_graph(int(n * .4), int(n * .3))
        return readwrite._wrap(func(5))

    def test_send_network_no_domain(self):
        """
        Some networks do not have a domain.
        GH-59
        """
        self.send_signal(self.widget.Inputs.network, self._create_graph())

    def test_select_nodes(self):
        """
        Do not fail on a wrong subset data and a graph without items.
        GH-67
        """
        w = self.widget
        self.send_signal(w.Inputs.network, self._create_graph())
        table = Table("iris")
        self.send_signal(w.Inputs.node_subset, table)
        w.set_selection_mode()

    def test_show_edge_weights(self):
        """
        Do not crash when there is no graph and
        one click on Show edge weights.
        GH-68
        """
        self.widget.checkbox_show_weights.click()
