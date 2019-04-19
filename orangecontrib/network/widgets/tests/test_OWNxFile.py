import os
import unittest
from unittest.mock import patch, Mock

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxFile import OWNxFile
import orangecontrib.network


class TestOWNxFile(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxFile)  # type: OWNxFile

    def test_read_error(self):
        with patch("orangecontrib.network.widgets.OWNxFile.read_pajek",
                   Mock(side_effect=OSError)):
            self.widget.open_net_file("foo.net")
        self.assertTrue(self.widget.Error.io_error.is_shown())
        filename = os.path.join(
            os.path.split(orangecontrib.network.__file__)[0],
            "networks/leu_by_genesets.net")
        self.widget.open_net_file(filename)
        self.assertFalse(self.widget.Error.io_error.is_shown())


if __name__ == "__main__":
    unittest.main()
