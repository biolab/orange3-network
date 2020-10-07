import unittest
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp

from Orange.data import Domain, DiscreteVariable, ContinuousVariable, Table

from orangecontrib.network import Network
from orangecontrib.network.widgets.ownxsinglemode import OWNxSingleMode
from orangecontrib.network.widgets.tests.utils import NetworkTest


class TestOWNxSingleMode(NetworkTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxSingleMode)  # type: OWNxSingleMode

        self.a, self.b, self.c = [
            DiscreteVariable("a", values=("a0", "a1")),
            ContinuousVariable("b"),
            DiscreteVariable("c", values=("c0", "c1", "c2", "c3"))]
        self.domain = Domain([self.a, self.b, self.c])
        self.table = Table.from_numpy(self.domain, np.array([
            [0, 0, 2],
            [1, 0, 3],
            [1, 0, 1],
            [0, 2, 1],
            [0, 0, 1]
        ]))

        self.d = DiscreteVariable("d", values=("d0", ))

    def _set_graph(self, data, edges=None):
        net = Network(
            data,
            sp.csr_matrix((len(data), len(data))) if edges is None else edges)
        self.send_signal(self.widget.Inputs.network, net)

    def test_combo_inits(self):
        widget = self.widget
        model = widget.controls.variable.model()
        cb_select = widget.controls.connect_value
        cb_connect = widget.controls.connector_value

        self.assertSequenceEqual(model, [])
        self.assertIsNone(widget.variable)
        self.assertEqual(cb_select.count(), 0)
        self.assertEqual(cb_connect.count(), 0)
        self.assertFalse(widget.Error.no_data.is_shown())
        self.assertFalse(widget.Error.no_categorical.is_shown())
        self.assertFalse(widget.Error.same_values.is_shown())

        self._set_graph(self.table)
        model = widget.controls.variable.model()
        self.assertSequenceEqual(model, [self.a, self.c])
        self.assertIs(widget.variable, self.a)
        self.assertEqual(cb_select.count(), 2)
        self.assertEqual(cb_select.itemText(0), "a0")
        self.assertEqual(cb_select.itemText(1), "a1")
        self.assertEqual(cb_connect.count(), 3)
        self.assertEqual(cb_connect.itemText(0), "(all others)")
        self.assertEqual(cb_connect.itemText(1), "a0")
        self.assertEqual(cb_connect.itemText(2), "a1")

        self.send_signal(widget.Inputs.network, None)
        self.assertSequenceEqual(model, [])
        self.assertIsNone(widget.variable)
        self.assertEqual(cb_select.count(), 0)
        self.assertEqual(cb_connect.count(), 0)
        self.assertFalse(widget.Error.no_data.is_shown())
        self.assertFalse(widget.Error.no_categorical.is_shown())
        self.assertFalse(widget.Error.same_values.is_shown())

        self._set_graph(Table.from_domain(Domain([], [], [self.a, self.c])))
        self.assertSequenceEqual(model, [self.a, self.c])
        self.assertIs(widget.variable, self.a)

    def test_no_single_valued_vars(self):
        self._set_graph(Table.from_domain(Domain([self.a, self.b, self.c, self.d])))

    def test_show_errors(self):
        widget = self.widget
        model = widget.controls.variable.model()
        a, b, c, d = self.a, self.b, self.c, self.d
        cb_connector = widget.controls.connector_value

        no_data = widget.Error.no_data.is_shown
        no_categorical = widget.Error.no_categorical.is_shown
        same_values = widget.Error.same_values.is_shown

        self._set_graph(Table.from_domain(Domain([a, b, c, d])))
        self.assertSequenceEqual(model, [a, c])
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table.from_domain(Domain([b, d])))
        self.assertSequenceEqual(model, [])
        self.assertFalse(no_data())
        self.assertTrue(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table.from_domain(Domain([a, b, c, d])))
        self.assertSequenceEqual(model, [a, c])
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        widget.connector_value = widget.connect_value + 1
        cb_connector.activated[int].emit(widget.connector_value)
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertTrue(same_values())

        net = Network(range(3), sp.csr_matrix([[0, 1], [1, 2]]))
        self.send_signal(widget.Inputs.network, net)
        self.assertTrue(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table.from_domain(Domain([a, b, c, d])))
        widget.connector_value = widget.connect_value + 1
        self.send_signal(widget.Inputs.network, None)
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table.from_domain(Domain([a, b, c, d])))
        widget.connector_value = widget.connect_value + 1
        cb_connector.activated[int].emit(widget.connector_value)
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertTrue(same_values())

        self._set_graph(Table.from_domain(Domain([b, d])))
        self.assertFalse(no_data())
        self.assertTrue(no_categorical())
        self.assertFalse(same_values())

    def test_value_combo_updates(self):
        widget = self.widget
        widget.update_output = Mock()
        cb_kept = widget.controls.connect_value
        a, c = self.a, self.c

        self._set_graph(Table.from_domain(Domain([a, c])))
        self.assertEqual(len(cb_kept), 2)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

        widget.variable = c
        widget.controls.variable.activated[int].emit(1)
        self.assertEqual(len(cb_kept), 4)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

        widget.connect_value = 3
        widget.variable = a
        widget.controls.variable.activated[int].emit(0)
        self.assertEqual(len(cb_kept), 2)
        self.assertEqual(widget.connect_value, 0)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

    def test_disable_combo_on_binary(self):
        widget = self.widget
        a, c = self.a, self.c
        cb_variable = widget.controls.variable
        cb_connect = widget.controls.connect_value
        cb_connector = widget.controls.connector_value

        self.assertFalse(cb_connector.isEnabled())

        self._set_graph(Table.from_domain(Domain([a, c])))
        self.assertFalse(cb_connector.isEnabled())
        self.assertEqual(widget.connector_value, 2)

        widget.connect_value = 1
        cb_connect.activated[int].emit(1)
        self.assertFalse(cb_connector.isEnabled())
        self.assertEqual(widget.connector_value, 1)

        widget.variable = c
        cb_variable.activated[int].emit(1)
        self.assertTrue(cb_connector.isEnabled())
        self.assertEqual(widget.connect_value, 0)
        self.assertEqual(widget.connector_value, 0)

        widget.variable = a
        cb_variable.activated[int].emit(0)
        self.assertFalse(cb_connector.isEnabled())
        self.assertEqual(widget.connect_value, 0)
        self.assertEqual(widget.connector_value, 2)

    def test_callbacks_called_on_value(self):
        widget = self.widget
        send = widget.Outputs.network.send = Mock()
        update = widget.update_output = Mock(side_effect=widget.update_output)

        self._set_graph(Table.from_domain(Domain([self.c])))
        update.assert_called()
        update.reset_mock()
        send.assert_called()
        send.reset_mock()

        widget.connect_value = 1
        widget.controls.connect_value.activated[int].emit(1)
        update.assert_called()
        update.reset_mock()
        send.assert_called()
        send.reset_mock()

        widget.connector_value = 1
        widget.controls.connector_value.activated[int].emit(1)
        update.assert_called()
        update.reset_mock()
        send.assert_called()
        send.reset_mock()

    @patch("orangecontrib.network.network.twomode.to_single_mode")
    def test_masks_in_update(self, to_single_mode):
        widget = self.widget

        def check_call(expected_mode_mask, expected_conn_mask):
            to_single_mode.assert_called()
            net, mode_mask, conn_mask, weighting = to_single_mode.call_args[0]
            self.assertIs(net, widget.network)
            np.testing.assert_almost_equal(mode_mask, expected_mode_mask)
            np.testing.assert_almost_equal(conn_mask, expected_conn_mask)
            self.assertEqual(weighting, widget.weighting)

        self._set_graph(self.table)

        widget.variable = self.a
        widget.connect_value = 0
        widget.connector_value = 0
        widget.update_output()
        check_call(
            np.array([True, False, False, True, True]),
            np.array([False, True, True, False, False]))
        widget.connector_value = 2
        widget.update_output()
        check_call(
            np.array([True, False, False, True, True]),
            np.array([False, True, True, False, False]))
        widget.connect_value = 1
        widget.connector_value = 0
        widget.update_output()
        check_call(
            np.array([False, True, True, False, False]),
            np.array([True, False, False, True, True]))
        widget.connector_value = 2
        widget.update_output()
        check_call(
            np.array([False, True, True, False, False]),
            np.array([True, False, False, True, True]))

        widget.variable = self.c
        widget.connect_value = 0
        widget.connector_value = 0
        widget.update_output()
        check_call(
            np.array([False] * 5),
            np.array([True] * 5))

        widget.connect_value = 1
        widget.connector_value = 0
        widget.update_output()
        check_call(
            np.array([False, False, True, True, True]),
            np.array([True, True, False, False, False]))

        widget.connect_value = 1
        widget.connector_value = 1
        widget.update_output()
        check_call(
            np.array([False, False, True, True, True]),
            np.array([False, False, False, False, False]))

        widget.connect_value = 1
        widget.connector_value = 3
        widget.update_output()
        check_call(
            np.array([False, False, True, True, True]),
            np.array([True, False, False, False, False]))

        widget.connect_value = 1
        widget.connector_value = 4
        widget.update_output()
        check_call(
            np.array([False, False, True, True, True]),
            np.array([False, True, False, False, False]))

        widget.connect_value = 2
        widget.connector_value = 0
        widget.update_output()
        check_call(
            np.array([True, False, False, False, False]),
            np.array([False, True, True, True, True]))

        widget.connect_value = 2
        widget.connector_value = 1
        widget.update_output()
        check_call(
            np.array([True, False, False, False, False]),
            np.array([False, False, False, False, False]))

        widget.connect_value = 2
        widget.connector_value = 2
        widget.update_output()
        check_call(
            np.array([True, False, False, False, False]),
            np.array([False, False, True, True, True]))

        widget.connect_value = 2
        widget.connector_value = 4
        widget.update_output()
        check_call(
            np.array([True, False, False, False, False]),
            np.array([False, True, False, False, False]))

        widget.weighting = 3
        widget.update_output()
        check_call(
            np.array([True, False, False, False, False]),
            np.array([False, True, False, False, False]))

    def test_send_report(self):
        self._set_graph(self.table)
        self.widget.send_report()

    def test_missing_data(self):
        widget = self.widget
        network = self._read_network("davis.net")
        num_total = len(network.nodes)
        network.nodes.X[0, 1] = np.nan  # hide a node's role (= person in this case)
        self.send_signal(widget.Inputs.network, network)

        # Feature: role, Connect: person
        widget.variable = network.nodes.domain.attributes[1]
        widget.connect_value = 1
        widget.connector_value = 1
        widget.update_output()
        self.assertTrue(widget.Warning.ignoring_missing.is_shown())
        person_network = self.get_output(widget.Outputs.network)
        num_persons = len(person_network.nodes)

        # Feature: role, Connect: event
        widget.connect_value = 0
        widget.connector_value = 2
        widget.update_output()
        event_network = self.get_output(widget.Outputs.network)
        num_events = len(event_network.nodes)

        # Make sure the node with missing data is ignored for both modes
        self.assertEqual(num_persons + num_events, num_total - 1)


if __name__ == "__main__":
    unittest.main()
