import networkx as nx

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxExplorer import OWNxExplorer
from orangecontrib.network import readwrite


class TestOWGradientDescent(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWNxExplorer)  # type: OWNxExplorer

    def test_send_network_no_domain(self):
        """
        Some networks do not have a domain.
        GH-59
        """
        w = self.widget
        func = lambda n: nx.barbell_graph(int(n * .4), int(n * .3))
        graph = readwrite._wrap(func(5))
        self.send_signal(w.Inputs.network, graph)

    def test_show_edge_weights(self):
        """
        Do not crash when there is no graph and
        one click on Show edge weights.
        GH-68
        """
        self.widget.checkbox_show_weights.click()
