import os
import unittest
from unittest.mock import patch, Mock

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxFile import OWNxFile
import orangecontrib.network

class TestOWNxExplorer(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxFile)  # type: OWNxFile

    def test_read_error(self):
        with patch("orangecontrib.network.readwrite.read",
                   Mock(side_effect=ValueError)):
            self.widget.openNetFile("foo.net")
        self.assertTrue(self.widget.Error.error_reading_file.is_shown())
        filename = os.path.join(
            os.path.split(orangecontrib.network.__file__)[0],
            "networks/leu_by_genesets.net")
        self.widget.openNetFile(filename)
        self.assertFalse(self.widget.Error.error_reading_file.is_shown())


if __name__ == "__main__":
    unittest.main()
