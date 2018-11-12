import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import Domain, DiscreteVariable, ContinuousVariable, Table
from Orange.widgets.tests.base import WidgetTest

import orangecontrib.network
from orangecontrib.network.widgets.OWNxSingleMode import OWNxSingleMode


class TestOWNxSingleMode(WidgetTest):
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

        self.d = DiscreteVariable("d", values=["d0"])

    def _set_graph(self, data, edges=None):
        net = orangecontrib.network.Graph()
        net.add_nodes_from(range(len(data)))
        if edges is not None:
            net.add_edges_from(edges)
        net.set_items(data)
        self.send_signal(self.widget.Inputs.network, net)


class TestOWNxExplorerComputation(TestOWNxSingleMode):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWNxSingleMode)  # type: OWNxSingleMode

    def _set_graph(self, data, edges=None):
        net = orangecontrib.network.Graph()
        net.add_nodes_from(range(len(data)))
        if edges is not None:
            net.add_edges_from(edges)
        net.set_items(data)
        self.send_signal(self.widget.Inputs.network, net)

    def test_filtered_data_edges(self):
        def assertEdges(expected):
            self.assertEqual(set(map(tuple, fedges.tolist())), expected)

        widget = self.widget
        table = self.table
        self._set_graph(table, [(0, 1), (1, 3)])

        widget.variable = self.a
        widget.connect_value = 0
        widget.connector_value = 0
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[0, 0, 2], [0, 2, 1], [0, 0, 1]])
        assertEdges({(0, 1), (1, 1)})

        widget.connect_value = 1
        widget.connector_value = 0
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 3], [1, 0, 1]])
        assertEdges({(0, 3), (0, 0)})

        self._set_graph(table, [(0, 1), (1, 3), (2, 0), (3, 0), (4, 1)])
        widget.variable = self.c
        widget.connect_value = 0
        widget.connector_value = 0
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(len(fdata), 0)
        self.assertEqual(len(fedges), 0)

        widget.connect_value = 1
        widget.connector_value = 0
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 1], [0, 2, 1], [0, 0, 1]])
        assertEdges({(1, 1), (0, 0), (1, 0), (2, 1)})

        widget.connect_value = 1
        widget.connector_value = 1
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 1], [0, 2, 1], [0, 0, 1]])
        self.assertEqual(len(fedges), 0)

        widget.connect_value = 1
        widget.connector_value = 3
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 1], [0, 2, 1], [0, 0, 1]])
        assertEdges({(0, 0), (1, 0)})

        widget.connect_value = 1
        widget.connector_value = 4
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 1], [0, 2, 1], [0, 0, 1]])
        assertEdges({(1, 1), (2, 1)})

        self._set_graph(table)
        for widget.variable in (self.a, self.c):
            for widget.connect_value in range(len(widget.variable.values)):
                for widget.connector_value in \
                        range(len(widget.variable.values)):
                    if widget.connect_value != widget.connector_value:
                        fdata, fedges = widget._filtered_data_edges()
                        self.assertEqual(len(fdata), 0)
                        self.assertEqual(len(fedges), 0)

    def test_no_intramode_connections(self):
        self._set_graph(self.table, [(0, 3), (3, 4)])
        _, fedges = self.widget._filtered_data_edges()
        self.assertEqual(len(fedges), 0)

    def test_edges_and_intersections(self):
        self.assertEqual(
            list(self.widget._edges_and_intersections(np.array([
                [0, 0], [0, 1], [0, 3], [1, 1], [1, 2], [1, 3],
                [2, 0], [3, 4]
            ]))),
            [(0, 1, {1, 3}), (0, 2, {0})]
        )

    def test_weighted_edges(self):
        widget = self.widget
        w = "weight"

        widget.weighting = widget.Weighting.NoWeights
        self.assertEqual(
            widget._weighted_edges(
                iter([(0, 1, {1, 2, 3}), (0, 2, {3}), (2, 5, {1, 2})]),
                None),
            [(0, 1), (0, 2), (2, 5)]
        )

        widget.weighting = widget.Weighting.Connections
        self.assertEqual(
            widget._weighted_edges(
                iter([(0, 1, {1, 2, 3}), (0, 2, {3}), (2, 5, {1, 2})]),
                None),
            [(0, 1, {w: 3}), (0, 2, {w: 1}), (2, 5, {w: 2})]
        )

        widget.weighting = widget.Weighting.WeightedConnections
        edges = widget._weighted_edges(
            iter([(0, 1, {1, 2, 3}), (0, 2, {3}), (2, 5, {1, 2})]),
            np.array([[0, 1], [3, 1], [5, 1], [4, 2], [2, 3], [1, 3]]))
        edges = {(n1, n2): ws[w] for n1, n2, ws in edges}
        expected = [(0, 1, 1 / 9 + 1 + 1 / 4),
                  (0, 2, 1 / 4),
                  (2, 5, 1 / 9 + 1)]
        self.assertEqual(len(edges), len(expected))
        for n1, n2, w in expected:
            self.assertAlmostEqual(w, edges[(n1, n2)])


class TestOWNxExplorerGui(TestOWNxSingleMode):
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

        self._set_graph(Table(Domain([], [], [self.a, self.c])))
        self.assertSequenceEqual(model, [self.a, self.c])
        self.assertIs(widget.variable, self.a)

    def test_no_single_valued_vars(self):
        self._set_graph(Table(Domain([self.a, self.b, self.c, self.d])))

    def test_show_errors(self):
        widget = self.widget
        model = widget.controls.variable.model()
        a, b, c, d = self.a, self.b, self.c, self.d
        cb_connector = widget.controls.connector_value

        no_data = widget.Error.no_data.is_shown
        no_categorical = widget.Error.no_categorical.is_shown
        same_values = widget.Error.same_values.is_shown

        self._set_graph(Table(Domain([a, b, c, d])))
        self.assertSequenceEqual(model, [a, c])
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table(Domain([b, d])))
        self.assertSequenceEqual(model, [])
        self.assertFalse(no_data())
        self.assertTrue(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table(Domain([a, b, c, d])))
        self.assertSequenceEqual(model, [a, c])
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        widget.connector_value = widget.connect_value + 1
        cb_connector.activated[int].emit(widget.connector_value)
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertTrue(same_values())

        net = orangecontrib.network.Graph()
        net.add_edges_from(([0, 1], [1, 2]))
        self.send_signal(widget.Inputs.network, net)
        self.assertTrue(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table(Domain([a, b, c, d])))
        widget.connector_value = widget.connect_value + 1
        self.send_signal(widget.Inputs.network, None)
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table(Domain([a, b, c, d])))
        widget.connector_value = widget.connect_value + 1
        cb_connector.activated[int].emit(widget.connector_value)
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertTrue(same_values())

        self._set_graph(Table(Domain([b, d])))
        self.assertFalse(no_data())
        self.assertTrue(no_categorical())
        self.assertFalse(same_values())

    def test_value_combo_updates(self):
        widget = self.widget
        widget.update_output = Mock()
        cb_kept = widget.controls.connect_value
        a, c = self.a, self.c

        self._set_graph(Table(Domain([a, c])))
        self.assertEqual(len(cb_kept), 2)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

        self.variable = c
        widget.controls.variable.activated[int].emit(1)
        self.assertEqual(len(cb_kept), 4)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

        widget.connect_value = 3
        self.variable = a
        widget.controls.variable.activated[int].emit(0)
        self.assertEqual(len(cb_kept), 2)
        self.assertEqual(widget.connect_value, 0)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

    def test_callbacks_called_on_value(self):
        widget = self.widget
        send = widget.Outputs.network.send = Mock()

        self._set_graph(Table(Domain([self.c])))
        send.assert_called()
        send.reset_mock()

        widget.connect_value = 1
        widget.controls.connect_value.activated[int].emit(1)
        send.assert_called()
        send.reset_mock()

        widget.connector_value = 1
        widget.controls.connector_value.activated[int].emit(1)
        send.assert_called()
        send.reset_mock()

    def test_send_report(self):
        self._set_graph(self.table)
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()

