import unittest
from unittest.mock import patch

import numpy as np

from AnyQt.QtTest import QSignalSpy
from AnyQt.QtWidgets import QLineEdit
from AnyQt.QtCore import QEvent, Qt
from AnyQt.QtGui import QKeyEvent

from orangewidget.tests.base import GuiTest
from Orange.misc import DistMatrix
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxFromDistances import OWNxFromDistances, \
    Histogram, QIntValidatorWithFixup


class TestOWNxFromDistances(WidgetTest):
    def setUp(self):
        self.widget: OWNxFromDistances = self.create_widget(OWNxFromDistances)

        # Put non-zero elements in the diagonal to check that the widget ignores them
        self.distances = DistMatrix(np.array([[0., 1, 2, 5, 10],
                                              [1, -1, 5, 5, 13],
                                              [2, 5, 2, 6, 13],
                                              [5, 5, 6, 3, 15],
                                              [10, 13, 13, 15, 0]]))

    @staticmethod
    def set_edit(edit, value):
        edit.setText(str(value))
        edit.textChanged.emit(str(value))
        edit.textEdited.emit(str(value))
        edit.returnPressed.emit()
        edit.editingFinished.emit()

    def _assert_controls_enabled(self, enabled):
        widget = self.widget
        self.assertEqual(widget.threshold_edit.isEnabled(), enabled)
        self.assertEqual(widget.edges_edit.isEnabled(), enabled)
        self.assertEqual(widget.density_edit.isEnabled(), enabled)

    def test_set_weird_matrix(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)
        self.assertEqual(widget.eff_distances, 10)
        self.assertIsNotNone(self.get_output(widget.Outputs.network))
        self._assert_controls_enabled(True)

        self.send_signal(widget.Inputs.distances, None)
        self.assertIsNone(self.get_output(widget.Outputs.network))
        self._assert_controls_enabled(False)

        self.send_signal(widget.Inputs.distances, self.distances)
        self.assertIsNotNone(self.get_output(widget.Outputs.network))
        self._assert_controls_enabled(True)

        self.send_signal(widget.Inputs.distances, DistMatrix(np.zeros((0, 0))))
        self.assertIsNone(self.get_output(widget.Outputs.network))
        self._assert_controls_enabled(False)

        self.send_signal(widget.Inputs.distances, DistMatrix(np.array([[1]])))
        self.assertIsNone(self.get_output(widget.Outputs.network))
        self._assert_controls_enabled(False)

        self.send_signal(widget.Inputs.distances, None)
        self.assertIsNone(self.get_output(widget.Outputs.network))
        self._assert_controls_enabled(False)

        self.send_signal(widget.Inputs.distances,
                         DistMatrix(np.array([[0, 1],
                                              [1, 0]])))
        self.assertIsNotNone(self.get_output(widget.Outputs.network))
        self._assert_controls_enabled(True)

    @patch("orangecontrib.network.widgets.OWNxFromDistances.Histogram.set_graph")
    def test_compute_histogram_symmetric(self, set_graph):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)
        np.testing.assert_almost_equal(widget.thresholds, [1, 2, 5, 6, 10, 13, 15])
        np.testing.assert_almost_equal(widget.cumfreqs, [1, 2, 5, 6, 7, 9, 10])
        set_graph.assert_called_with(widget.thresholds, widget.cumfreqs)

        # The matrix is symmetric, so the number of bins is below 1000 ->
        # the histogram is computed exactly
        distances = np.zeros((40, 40))
        for i in range(40):
            distances[:i, i] = distances[i, :i] = np.arange(i)
        distances = DistMatrix(distances)
        self.send_signal(widget.Inputs.distances, distances)
        np.testing.assert_almost_equal(widget.thresholds, np.arange(39))
        set_graph.assert_called_with(widget.thresholds, widget.cumfreqs)

        # Even though the matrix is symmetric, the number of bins is above 1000
        distances = np.zeros((50, 50))
        for i in range(50):
            distances[:i, i] = distances[i, :i] = np.arange(i)
        distances = DistMatrix(distances)
        self.send_signal(widget.Inputs.distances, distances)
        np.testing.assert_almost_equal(
            widget.thresholds[:5], [0.048, 0.096, 0.144, 0.192, 0.24])
        set_graph.assert_called_with(widget.thresholds, widget.cumfreqs)

    @patch("orangecontrib.network.widgets.OWNxFromDistances.Histogram.set_graph")
    def test_compute_histogram_asymmetric(self, set_graph):
        widget = self.widget

        self.distances[0, 1] = 1.5
        self.send_signal(widget.Inputs.distances, self.distances)
        np.testing.assert_almost_equal(widget.thresholds, [1, 1.5, 2, 5, 6, 10, 13, 15])
        np.testing.assert_almost_equal(widget.cumfreqs, [1, 2, 4, 10, 12, 14, 18, 20])
        set_graph.assert_called_with(widget.thresholds, widget.cumfreqs)

        distances = DistMatrix(np.array(np.arange(40 * 40).reshape((40, 40))))
        self.send_signal(widget.Inputs.distances, distances)
        np.testing.assert_almost_equal(
            widget.thresholds[:5], [2.597, 4.194, 5.791, 7.388, 8.985])
        np.testing.assert_almost_equal(
            widget.cumfreqs[:5], [2, 4, 5, 7, 8])
        self.assertEqual(widget.cumfreqs[-1], 40 * 39)
        set_graph.assert_called_with(widget.thresholds, widget.cumfreqs)

    def test_set_symmetric(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)
        self.assertTrue(widget.symmetric)
        self.assertEqual(widget.eff_distances, 10)

        self.distances[0, 1] = 2
        self.send_signal(widget.Inputs.distances, self.distances)
        self.assertFalse(widget.symmetric)
        self.assertEqual(widget.eff_distances, 20)

    def test_set_threshold_from_density_symmetric(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for density, threshold in [(0, 0), (10, 1), (20, 2), (30, 5), (40, 5),
                                   (50, 5), (60, 6), (70, 10), (80, 13),
                                   (90, 13), (100, 15)]:
            self.set_edit(widget.density_edit, density)
            self.assertEqual(widget.threshold, threshold,
                             msg=f"at density={density}")

    def test_set_threshold_from_density_asymmetric(self):
        widget = self.widget
        self.distances[0, 1] = 2
        self.send_signal(widget.Inputs.distances, self.distances)

        for density, threshold in [(0, 0), (10, 2), (20, 2), (21, 5), (50, 5),
                                   (51, 6), (60, 6), (70, 10), (80, 13),
                                   (90, 13), (100, 15)]:
            self.set_edit(widget.density_edit, density)
            self.assertEqual(widget.threshold, threshold,
                             msg=f"at density={density}")

    def test_set_threshold_from_edges(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for edges, threshold in [(0, 0), (1, 1), (2, 2), (3, 5), (4, 5), (5, 5),
                                 (6, 6), (7, 10), (8, 13), (9, 13), (10, 15)]:
            self.set_edit(widget.edges_edit, edges)
            self.assertEqual(widget.threshold, threshold,
                             msg=f"at edges={edges}")

    def test_edges_from_threshold_symmetric(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for threshold, edges in [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2),
                                 (5, 5), (6, 6), (7, 6), (8, 6), (9, 6),
                                 (10, 7), (12.99, 7), (13, 9), (15, 10),
                                 (16, 10)]:
            self.set_edit(widget.threshold_edit, threshold)
            self.assertEqual(float(widget.edges_edit.text()), edges,
                             msg=f"at threshold={threshold}")

    def test_edges_from_threshold_asymmetric(self):
        widget = self.widget
        self.distances[0, 1] = 1.5
        self.send_signal(widget.Inputs.distances, self.distances)

        for threshold, edges in [(0, 0), (1, 1), (1.5, 2), (2, 4), (3, 4), (4, 4),
                                 (5, 10), (6, 12), (7, 12), (8, 12), (9, 12),
                                 (10, 14), (12.99, 14), (13, 18), (15, 20),
                                 (16, 20)]:
            self.set_edit(widget.threshold_edit, threshold)
            self.assertEqual(float(widget.edges_edit.text()), edges,
                             msg=f"at threshold={threshold}")

    def test_set_edges(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for edges, threshold, density in [(0, 0, 0), (1, 1, 10), (2, 2, 20),
                                          (3, 5, 50), (4, 5, 50), (5, 5, 50),
                                          (6, 6, 60), (7, 10, 70),
                                          (8, 13, 90), (9, 13, 90), (10, 15, 100)]:
            self.set_edit(widget.edges_edit, edges)
            self.assertEqual(widget.threshold, threshold,
                             msg=f"at edges={edges}")
            self.assertEqual(widget.edges_edit.text(), str(density // 10),
                             msg=f"at edges={edges}")
            self.assertEqual(widget.density, density,
                             msg=f"at edges={edges}")
            self.assertEqual(widget.histogram.hline.value(), density // 10,
                             msg=f"at edges={edges}")
            self.assertEqual(widget.histogram.vline.value(),
                             min(threshold, np.max(self.distances)),
                             msg=f"at edges={edges}")

    def test_set_density(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for density, threshold, edges in [(0, 0, 0), (10, 1, 1), (20, 2, 2),
                                          (30, 5, 5), (40, 5, 5), (50, 5, 5),
                                          (60, 6, 6), (70, 10, 7),
                                          (80, 13, 9), (90, 13, 9), (100, 15, 10)]:
            self.set_edit(widget.density_edit, density)
            self.assertEqual(widget.threshold, threshold,
                             msg=f"at density={density}")
            self.assertEqual(int(widget.edges_edit.text()), edges,
                             msg=f"at density={density}")
            self.assertEqual(widget.density, edges * 10)
            self.assertEqual(widget.histogram.hline.value(), edges,
                             msg=f"at threshold={threshold}")
            self.assertEqual(widget.histogram.vline.value(),
                             min(threshold, np.max(self.distances)),
                             msg=f"at threshold={threshold}")

    def test_set_threshold(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for threshold, density, edges in [(0, 0, 0), (1, 10, 1), (2, 20, 2),
                                          (3, 20, 2), (4, 20, 2), (5, 50, 5),
                                          (6, 60, 6), (9, 60, 6), (10, 70, 7),
                                          (12.9, 70, 7), (13, 90, 9),
                                          (14.9, 90, 9), (15, 100, 10),
                                          (16, 100, 10)]:
            self.set_edit(widget.threshold_edit, threshold)
            self.assertEqual(widget.density, density,
                             msg=f"at threshold={threshold}")
            self.assertEqual(int(widget.edges_edit.text()), edges,
                             msg=f"at threshold={threshold}")
            self.assertEqual(widget.histogram.hline.value(), edges,
                             msg=f"at threshold={threshold}")
            self.assertEqual(widget.histogram.vline.value(),
                             min(threshold, np.max(self.distances)),
                             msg=f"at threshold={threshold}")

    @patch("orangecontrib.network.widgets.OWNxFromDistances.OWNxFromDistances.generate_network")
    def test_set_threshold_from_histogram(self, generate_network):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        generate_network.reset_mock()
        widget.histogram.thresholdChanged.emit(5)
        generate_network.assert_not_called()
        self.assertEqual(widget.threshold, 5)
        self.assertEqual(widget.density, 50)
        self.assertEqual(widget.edges_edit.text(), "~5")

        widget.histogram.thresholdChanged.emit(11)
        generate_network.assert_not_called()
        self.assertEqual(widget.threshold, 11)
        self.assertEqual(widget.density, 70)
        self.assertEqual(widget.edges_edit.text(), "~7")

        widget.histogram.thresholdChanged.emit(15)
        generate_network.assert_not_called()
        self.assertEqual(widget.threshold, 15)
        self.assertEqual(widget.density, 100)
        self.assertEqual(widget.edges_edit.text(), "~10")

        widget.histogram.thresholdChanged.emit(16)
        generate_network.assert_not_called()
        self.assertEqual(widget.threshold, 16)
        self.assertEqual(widget.density, 100)
        self.assertEqual(widget.edges_edit.text(), "~10")

        widget.histogram.draggingFinished.emit()
        generate_network.assert_called_once()

    def test_generate_network(self):
        widget = self.widget

        distances = DistMatrix(np.array([[0., 1, 2, 5, 10],
                                         [1, -1, 5, 5, 3],
                                         [2, 5, 2, 6, 13],
                                         [5, 5, 6, 3, 15],
                                         [10, 3, 13, 15, 0]]))

        widget.threshold = 2
        self.send_signal(widget.Inputs.distances, distances)
        graph = self.get_output(widget.Outputs.network)
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 2)
        coo = graph.edges[0].edges.tocoo()
        np.testing.assert_equal(coo.row, [1, 2])
        np.testing.assert_equal(coo.col, [0, 0])
        np.testing.assert_almost_equal(coo.data, [1 / 10 ** 0.5, 0.1])
        self.assertEqual(list(graph.nodes), list("12345"))

        self.set_edit(widget.threshold_edit, 3)
        graph = self.get_output(widget.Outputs.network)
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 3)
        coo = graph.edges[0].edges.tocoo()
        np.testing.assert_equal(coo.row, [1, 2, 4])
        np.testing.assert_equal(coo.col, [0, 0, 1])
        np.testing.assert_almost_equal(coo.data, [1 / 10 ** (1 / 3), 1 / 10 ** (2 / 3), 0.1])

        self.set_edit(widget.threshold_edit, 4)
        graph = self.get_output(widget.Outputs.network)
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 3)
        coo = graph.edges[0].edges.tocoo()
        np.testing.assert_equal(coo.row, [1, 2, 4])
        np.testing.assert_equal(coo.col, [0, 0, 1])
        np.testing.assert_almost_equal(coo.data, [1 / 10 ** (1 / 3), 1 / 10 ** (2 / 3), 0.1])

        self.set_edit(widget.threshold_edit, 5)
        graph = self.get_output(widget.Outputs.network)
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 6)
        coo = graph.edges[0].edges.tocoo()
        np.testing.assert_equal(coo.row, [1, 2, 2, 3, 3, 4])
        np.testing.assert_equal(coo.col, [0, 0, 1, 0, 1, 1])
        np.testing.assert_almost_equal(coo.data, [1 / 10 ** (1 / 5), 1 / 10 ** (2 / 5),
                                                  0.1, 0.1, 0.1, 1 / 10 ** (3 / 5)])

    def test_threshold_decimals(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        widget.threshold = 1.23412
        widget.update_edits(widget.edges_edit)
        self.assertEqual(widget.threshold_edit.text(), "1.2")

        self.send_signal(widget.Inputs.distances, self.distances / 100)
        widget.threshold = 1.23412
        widget.update_edits(widget.edges_edit)
        self.assertEqual(widget.threshold_edit.text(), "1.234")

    def test_too_many_edges(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        with patch("orangecontrib.network.widgets.OWNxFromDistances."
                   "OWNxFromDistances.edges_from_threshold",
                   return_value=1_000_000):
            widget.generate_network()
        self.assertTrue(widget.Error.number_of_edges.is_shown())

        widget.generate_network()
        self.assertFalse(widget.Error.number_of_edges.is_shown())

    def test_report(self):
        widget = self.widget
        widget.send_report()

        self.send_signal(widget.Inputs.distances, self.distances)
        widget.send_report()


class TestHistogram(GuiTest):
    def setUp(self):
        self.histogram = Histogram(None)
        assert set(self.histogram._elements()) == \
               {self.histogram.hline, self.histogram.vline,
                self.histogram.curve, self.histogram.fill_curve}

    def test_show_hide_elements(self):
        hist = self.histogram

        self.assertFalse(any(x.isVisible() for x in hist._elements()))
        hist.set_graph(np.array([1, 2, 5, 6, 10, 13, 15]),
                       np.array([1, 2, 5, 6, 7, 9, 10]))
        self.assertTrue(all(x.isVisible() for x in hist._elements()))

        hist.clear_graph()
        self.assertFalse(any(x.isVisible() for x in hist._elements()))

    def test_ranges(self):
        hist = self.histogram

        hist.set_graph(np.array([1, 2, 5, 6, 10, 13, 15]),
                       np.array([1, 2, 5, 6, 7, 9, 10]))
        self.assertEqual(hist.hline.bounds(), [0, 10])
        self.assertEqual(hist.vline.bounds(), [0, 15])
        self.assertEqual(hist.getAxis("left").range, [0, 10])
        self.assertEqual(hist.getAxis("bottom").range, [0, 15])
        self.assertAlmostEqual(hist.prop_axis.scale, 1 / 10 * 100)

    def _assert_fill_curve(self, hist, data):
        for exp, act in zip(data, hist.fill_curve.getData()):
            np.testing.assert_almost_equal(exp, act)

    def test_drag_hline(self):
        hist = self.histogram
        spy = QSignalSpy(hist.thresholdChanged)

        hist.set_graph(np.array([1, 2, 5, 6, 10, 13, 15]),
                       np.array([1, 2, 5, 6, 7, 9, 10]))

        hist.hline.setPos(1)
        hist.hline.sigDragged.emit(hist.hline)
        self.assertEqual(hist.vline.value(), 1)
        self.assertEqual(list(spy), [[1]])
        self._assert_fill_curve(hist, ([0, 1], [0, 1]))

        hist.hline.setPos(1.5)
        hist.hline.sigDragged.emit(hist.hline)
        self.assertEqual(hist.vline.value(), 2)
        self.assertEqual(list(spy), [[1], [2]])
        self._assert_fill_curve(hist, ([0, 1, 2], [0, 1, 2]))

        hist.hline.setPos(6.8)
        hist.hline.sigDragged.emit(hist.hline)
        self.assertEqual(hist.vline.value(), 10)
        self.assertEqual(list(spy), [[1], [2], [10]])
        self._assert_fill_curve(hist,
                                ([0, 1, 2, 5, 6, 10],
                                 [0, 1, 2, 5, 6, 7]))

        hist.hline.setPos(7)
        hist.hline.sigDragged.emit(hist.hline)
        self.assertEqual(hist.vline.value(), 10)
        self.assertEqual(list(spy), [[1], [2], [10], [10]])
        self._assert_fill_curve(hist,
                                ([0, 1, 2, 5, 6, 10],
                                 [0, 1, 2, 5, 6, 7]))

        hist.hline.setPos(7.2)
        hist.hline.sigDragged.emit(hist.hline)
        self.assertEqual(hist.vline.value(), 13)
        self.assertEqual(list(spy), [[1], [2], [10], [10], [13]])
        self._assert_fill_curve(hist,
                                ([0, 1, 2, 5, 6, 10, 13],
                                 [0, 1, 2, 5, 6, 7, 9]))

        hist.hline.setPos(10)
        hist.hline.sigDragged.emit(hist.hline)
        self.assertEqual(hist.vline.value(), 15)
        self.assertEqual(list(spy), [[1], [2], [10], [10], [13], [15]])
        self._assert_fill_curve(hist,
                                ([0, 1, 2, 5, 6, 10, 13, 15],
                                 [0, 1, 2, 5, 6, 7, 9, 10]))

        hist.hline.setPos(0)
        hist.hline.sigDragged.emit(hist.hline)
        self.assertEqual(hist.vline.value(), 0)
        self.assertEqual(list(spy), [[1], [2], [10], [10], [13], [15], [0]])
        self._assert_fill_curve(hist,
                                ([0],
                                 [0]))

    def test_drag_vline(self):
        hist = self.histogram
        spy = QSignalSpy(hist.thresholdChanged)

        hist.set_graph(np.array([1, 2, 5, 6, 10, 13, 15]),
                       np.array([1, 2, 5, 6, 7, 9, 10]))

        hist.vline.setPos(1)
        hist.vline.sigDragged.emit(hist.vline)
        self.assertEqual(hist.hline.value(), 1)
        self.assertEqual(list(spy), [[1]])
        self._assert_fill_curve(hist, ([0, 1], [0, 1]))

        hist.vline.setPos(1.5)
        hist.vline.sigDragged.emit(hist.vline)
        self.assertEqual(hist.hline.value(), 1)
        self.assertEqual(list(spy), [[1], [1.5]])
        self._assert_fill_curve(hist, ([0, 1], [0, 1]))

        hist.vline.setPos(2)
        hist.vline.sigDragged.emit(hist.vline)
        self.assertEqual(hist.hline.value(), 2)
        self.assertEqual(list(spy), [[1], [1.5], [2]])
        self._assert_fill_curve(hist, ([0, 1, 2], [0, 1, 2]))

        hist.vline.setPos(9)
        hist.vline.sigDragged.emit(hist.vline)
        self.assertEqual(hist.hline.value(), 6)
        self.assertEqual(list(spy), [[1], [1.5], [2], [9]])
        self._assert_fill_curve(hist,
                                ([0, 1, 2, 5, 6],
                                 [0, 1, 2, 5, 6]))

        hist.vline.setPos(10)
        hist.vline.sigDragged.emit(hist.vline)
        self.assertEqual(hist.hline.value(), 7)
        self.assertEqual(list(spy), [[1], [1.5], [2], [9], [10]])
        self._assert_fill_curve(hist,
                                ([0, 1, 2, 5, 6, 10],
                                 [0, 1, 2, 5, 6, 7]))

        hist.vline.setPos(10.5)
        hist.vline.sigDragged.emit(hist.vline)
        self.assertEqual(hist.hline.value(), 7)
        self.assertEqual(list(spy), [[1], [1.5], [2], [9], [10], [10.5]])
        self._assert_fill_curve(hist,
                                ([0, 1, 2, 5, 6, 10],
                                 [0, 1, 2, 5, 6, 7]))

        hist.vline.setPos(13)
        hist.vline.sigDragged.emit(hist.vline)
        self.assertEqual(hist.hline.value(), 9)
        self.assertEqual(list(spy), [[1], [1.5], [2], [9], [10], [10.5], [13]])
        self._assert_fill_curve(hist,
                                ([0, 1, 2, 5, 6, 10, 13],
                                 [0, 1, 2, 5, 6, 7, 9]))

        hist.vline.setPos(0)
        hist.vline.sigDragged.emit(hist.vline)
        self.assertEqual(hist.hline.value(), 0)
        self.assertEqual(list(spy), [[1], [1.5], [2], [9], [10], [10.5], [13], [0]])
        self._assert_fill_curve(hist,
                                ([0],
                                 [0]))


class TestQIntValidatorWithFixup(GuiTest):
    def test_validator(self):
        def enter_text(t):
            e.setText(t)
            e.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Enter, Qt.NoModifier))

        e = QLineEdit()
        e.setValidator(QIntValidatorWithFixup(0, 100, e))
        enter_text("")
        self.assertEqual(e.text(), "")
        enter_text("1")
        self.assertEqual(e.text(), "1")
        enter_text("100")
        self.assertEqual(e.text(), "100")
        enter_text("101")
        self.assertEqual(e.text(), "100")


if __name__ == "__main__":
    unittest.main()
