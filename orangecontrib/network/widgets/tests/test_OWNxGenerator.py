from orangecontrib.network.widgets.OWNxGenerator import OWNxGenerator
from orangecontrib.network.widgets.tests.utils import NetworkTest


class TestOWNxGenerator(NetworkTest):

    def setUp(self):
        self.widget = self.create_widget(OWNxGenerator)  # type: OWNxGenerator
