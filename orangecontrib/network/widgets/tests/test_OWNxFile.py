import os
import unittest
from unittest.mock import patch, Mock

import numpy as np

import Orange
from orangecontrib.network.widgets.OWNxFile import OWNxFile
from orangecontrib.network.widgets.tests.utils import NetworkTest

TEST_NETS = os.path.join(os.path.split(__file__)[0], "networks")

def _get_test_net(filename):
    return os.path.join(TEST_NETS, filename)


class TestOWNxFile(NetworkTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxFile)  # type: OWNxFile

    def test_read_error(self):
        with patch("orangecontrib.network.widgets.OWNxFile.read_pajek",
                   Mock(side_effect=OSError)):
            self.widget.open_net_file("foo.net")
        self.assertTrue(self.widget.Error.io_error.is_shown())
        filename = self._get_filename("leu_by_genesets.net")
        self.widget.open_net_file(filename)
        self.assertFalse(self.widget.Error.io_error.is_shown())

    def test_load_datafile(self):
        self.widget.open_net_file(_get_test_net("test.net"))
        items = self.get_output(self.widget.Outputs.items)
        self.assertEqual(items[0]["name"], "aaa")

    def test_invalid_datafile_length(self):
        # When data file's length does not match, the widget must create
        # a table from node labels
        self.widget.open_net_file(_get_test_net("test_inv.net"))
        self.assertTrue(self.widget.Error.mismatched_lengths)

        network = self.get_output(self.widget.Outputs.network)
        self.assertEqual(network.number_of_nodes(), 7)

        items = self.get_output(self.widget.Outputs.items)
        self.assertEqual(len(items), 7)
        self.assertEqual(items[0]["node_label"], "aa")

    def test_vars_for_label(self):
        self.widget.open_net_file(self._get_filename(None, mode="t"))

        data = Orange.data.Table(_get_test_net("test_data.tab"))
        domain = data.domain
        best_var, useful_vars = self.widget._vars_for_label(data)
        self.assertIs(best_var, domain["label"])
        self.assertEqual(useful_vars, [domain["with_extras"], domain["label"]])

    def test_label_combo_contents(self):
        widget = self.widget
        widget.read_auto_data = Mock()

        widget.open_net_file(self._get_filename(None, mode="t"))
        self.assertEqual(list(widget.label_model), [None])

        data = Orange.data.Table(_get_test_net("test_data.tab"))
        domain = data.domain

        self.send_signal(widget.Inputs.items, data)

        # Model contains useful variables
        self.assertEqual(list(widget.label_model),
                         [None, domain["with_extras"], domain["label"]])

        # `label` is chosen as default, and output has corresponding data
        self.assertIs(widget.label_variable, domain["label"])
        output = self.get_output(widget.Outputs.network)
        id_col, _ = output.nodes.get_column_view("id")
        np.testing.assert_equal(id_col, np.arange(1, 8))

        # No variable, row matching. Error is shown and original labels are used
        widget._label_to_tabel = Mock(return_value=data[:7])
        widget.label_variable = None
        widget.label_changed()
        self.assertTrue(widget.Error.mismatched_lengths.is_shown())
        output = self.get_output(widget.Outputs.network)
        self.assertIs(output.nodes, widget._label_to_tabel.return_value)

        # Choose a different variable; no error, output has corresponding data
        widget.label_variable = domain["with_extras"]
        widget.label_changed()
        self.assertFalse(widget.Error.mismatched_lengths.is_shown())
        output = self.get_output(widget.Outputs.network)
        id_col, _ = output.nodes.get_column_view("id")
        np.testing.assert_equal(id_col, np.arange(2, 9))

        # Remove data: model must be cleared, data back to original
        self.send_signal(widget.Inputs.items, None)
        self.assertEqual(list(widget.label_model), [None])
        output = self.get_output(widget.Outputs.network)
        self.assertIs(output.nodes, widget._label_to_tabel.return_value)

        # Bring data back, and turn on row matching;
        # this triggers an error; then remove data; error must be gone
        self.send_signal(widget.Inputs.items, data)
        widget.label_variable = None
        widget.label_changed()
        self.assertTrue(widget.Error.mismatched_lengths.is_shown())

        self.send_signal(widget.Inputs.items, None)
        output = self.get_output(widget.Outputs.network)
        self.assertIs(output.nodes, widget._label_to_tabel.return_value)
        self.assertFalse(widget.Error.mismatched_lengths.is_shown())

    def test_context_matching(self):
        widget = self.widget
        widget.open_net_file(self._get_filename(None, mode="t"))
        data = Orange.data.Table(_get_test_net("test_data.tab"))
        domain = data.domain

        self.send_signal(widget.Inputs.items, data)
        self.assertIs(widget.label_variable, domain["label"])

        widget.label_variable = domain["with_extras"]
        self.send_signal(widget.Inputs.items, None)
        self.assertIs(widget.label_variable, None)

        self.send_signal(widget.Inputs.items, data)
        self.assertIs(widget.label_variable, domain["with_extras"])


if __name__ == "__main__":
    unittest.main()
