from networkx.classes.graph import Graph

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxGenerator import OWNxGenerator, _balanced_tree, GraphType


class TestOWNxGenerator(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWNxGenerator)  # type: OWNxGenerator

    def test_balanced_tree(self):
        """
        Does new balanced tree work?
        GH-65
        """
        balanced_tree = _balanced_tree(100)
        self.assertIsInstance(balanced_tree, Graph)

    def test_regular_graph(self):
        """
        Random regular graph: inequality 0 <= d < n must be
        satisfied where d is degree of each node and n is
        the number of nodes.
        GH-69
        """
        w = self.widget
        w.controls.graph_type.setCurrentIndex(GraphType.all.index(GraphType.REGULAR))
        self.assertEqual(GraphType.all[w.controls.graph_type.currentIndex()][0], "Regular")
        w.graph_type = 12
        w.n_nodes = 1
        w.commit()
