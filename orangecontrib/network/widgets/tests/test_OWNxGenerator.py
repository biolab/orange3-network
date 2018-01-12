from networkx.classes.graph import Graph

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxGenerator import OWNxGenerator, _balanced_tree


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
