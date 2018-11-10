import unittest
from unittest.mock import patch, Mock

import numpy as np

from Orange.data import Domain, DiscreteVariable, ContinuousVariable, Table
from Orange.widgets.tests.base import WidgetTest

import orangecontrib.network
from orangecontrib.network.widgets.OWNxSingleMode import OWNxSingleMode


class TestOWNxExplorer(WidgetTest):
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

class TestOWNxExplorerComputation(TestOWNxExplorer):
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

        widget.mode_feature = self.a
        widget.kept_mode = 0
        widget.connecting_mode = 0
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[0, 0, 2], [0, 2, 1], [0, 0, 1]])
        assertEdges({(0, 1), (1, 1)})

        widget.kept_mode = 1
        widget.connecting_mode = 0
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 3], [1, 0, 1]])
        assertEdges({(0, 3), (0, 0)})

        self._set_graph(table, [(0, 1), (1, 3), (2, 0), (3, 0), (4, 1)])
        widget.mode_feature = self.c
        widget.kept_mode = 0
        widget.connecting_mode = 0
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(len(fdata), 0)
        self.assertEqual(len(fedges), 0)

        widget.kept_mode = 1
        widget.connecting_mode = 0
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 1], [0, 2, 1], [0, 0, 1]])
        assertEdges({(1, 1), (0, 0), (1, 0), (2, 1)})

        widget.kept_mode = 1
        widget.connecting_mode = 1
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 1], [0, 2, 1], [0, 0, 1]])
        self.assertEqual(len(fedges), 0)

        widget.kept_mode = 1
        widget.connecting_mode = 3
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 1], [0, 2, 1], [0, 0, 1]])
        assertEdges({(0, 0), (1, 0)})

        widget.kept_mode = 1
        widget.connecting_mode = 4
        fdata, fedges = widget._filtered_data_edges()
        self.assertEqual(list(fdata), [[1, 0, 1], [0, 2, 1], [0, 0, 1]])
        assertEdges({(1, 1), (2, 1)})

        self._set_graph(table)
        for widget.mode_feature in (self.a, self.c):
            for widget.kept_mode in range(len(widget.mode_feature.values)):
                for widget.connecting_mode in \
                        range(len(widget.mode_feature.values)):
                    if widget.kept_mode != widget.connecting_mode:
                        fdata, fedges = widget._filtered_data_edges()
                        self.assertEqual(len(fdata), 0)
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


class TestOWNxExplorerGui(TestOWNxExplorer):
    def test_combo_inits(self):
        widget = self.widget
        model = widget.controls.mode_feature.model()
        cb_select = widget.controls.kept_mode
        cb_connect = widget.controls.connecting_mode

        self.assertSequenceEqual(model, [])
        self.assertIsNone(widget.mode_feature)
        self.assertEqual(cb_select.count(), 0)
        self.assertEqual(cb_connect.count(), 0)
        self.assertFalse(widget.Error.no_data.is_shown())
        self.assertFalse(widget.Error.no_categorical.is_shown())
        self.assertFalse(widget.Error.same_values.is_shown())

        self._set_graph(self.table)
        model = widget.controls.mode_feature.model()
        self.assertSequenceEqual(model, [self.a, self.c])
        self.assertIs(widget.mode_feature, self.a)
        self.assertEqual(cb_select.count(), 2)
        self.assertEqual(cb_select.itemText(0), "a0")
        self.assertEqual(cb_select.itemText(1), "a1")
        self.assertEqual(cb_connect.count(), 3)
        self.assertEqual(cb_connect.itemText(0), "All")
        self.assertEqual(cb_connect.itemText(1), "a0")
        self.assertEqual(cb_connect.itemText(2), "a1")

        self.send_signal(widget.Inputs.network, None)
        self.assertSequenceEqual(model, [])
        self.assertIsNone(widget.mode_feature)
        self.assertEqual(cb_select.count(), 0)
        self.assertEqual(cb_connect.count(), 0)
        self.assertFalse(widget.Error.no_data.is_shown())
        self.assertFalse(widget.Error.no_categorical.is_shown())
        self.assertFalse(widget.Error.same_values.is_shown())

        self._set_graph(Table(Domain([], [], [self.a, self.c])))
        self.assertSequenceEqual(model, [self.a, self.c])
        self.assertIs(widget.mode_feature, self.a)

    def test_no_single_valued_vars(self):
        self._set_graph(Table(Domain([self.a, self.b, self.c, self.d])))

    def test_show_errors(self):
        widget = self.widget
        model = widget.controls.mode_feature.model()
        a, b, c, d = self.a, self.b, self.c, self.d
        cb_connecting = widget.controls.connecting_mode

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

        widget.connecting_mode = widget.kept_mode + 1
        cb_connecting.activated[int].emit(widget.connecting_mode)
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
        widget.connecting_mode = widget.kept_mode + 1
        self.send_signal(widget.Inputs.network, None)
        self.assertFalse(no_data())
        self.assertFalse(no_categorical())
        self.assertFalse(same_values())

        self._set_graph(Table(Domain([a, b, c, d])))
        widget.connecting_mode = widget.kept_mode + 1
        cb_connecting.activated[int].emit(widget.connecting_mode)
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
        cb_kept = widget.controls.kept_mode
        a, c = self.a, self.c

        self._set_graph(Table(Domain([a, c])))
        self.assertEqual(len(cb_kept), 2)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

        self.mode_feature = c
        widget.controls.mode_feature.activated[int].emit(1)
        self.assertEqual(len(cb_kept), 4)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

        widget.kept_mode = 3
        self.mode_feature = a
        widget.controls.mode_feature.activated[int].emit(0)
        self.assertEqual(len(cb_kept), 2)
        self.assertEqual(widget.kept_mode, 0)
        widget.update_output.assert_called()
        widget.update_output.reset_mock()

    def test_callbacks_called_on_mode(self):
        widget = self.widget
        send = widget.Outputs.network.send = Mock()

        self._set_graph(Table(Domain([self.c])))
        send.assert_called()
        send.reset_mock()

        widget.kept_mode = 1
        widget.controls.kept_mode.activated[int].emit(1)
        send.assert_called()
        send.reset_mock()

        widget.connecting_mode = 1
        widget.controls.connecting_mode.activated[int].emit(1)
        send.assert_called()
        send.reset_mock()

    def test_send_report(self):
        self._set_graph(self.table)
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()

