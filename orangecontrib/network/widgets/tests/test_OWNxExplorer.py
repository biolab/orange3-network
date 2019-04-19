from Orange.widgets.tests.base import WidgetTest
from orangecontrib.network.widgets.OWNxExplorer import OWNxExplorer


class TestOWNxExplorer(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxExplorer)  # type: OWNxExplorer

    def test_minimum_size(self):
        # Disable this test from the base test class
        pass
