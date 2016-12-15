from unittest import TestCase

from orangecontrib.network.widgets.OWNxTwitterGraph import OWNxTwitterGraph

class TestTwitterGraph(TestCase):

    def setUp(self):
        from PyQt4.QtGui import QApplication
        a = QApplication([])
        self.twitter = OWNxTwitterGraph()
        self.twitter.users.setText("orangedataminer")
        self.twitter.fetch_users()

    def test_output(self):
        self.assertGreater(self.twitter.n_followers, 0)
        self.assertGreater(self.twitter.n_following, 0)
        self.assertGreaterEqual(self.twitter.n_all, self.twitter.n_followers)
        self.assertGreaterEqual(self.twitter.n_all, self.twitter.n_following)
