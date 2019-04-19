from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxGenerator import OWNxGenerator


class TestOWNxGenerator(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWNxGenerator)  # type: OWNxGenerator

