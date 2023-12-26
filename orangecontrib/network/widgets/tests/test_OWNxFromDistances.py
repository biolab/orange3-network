import unittest
from unittest.mock import patch

import numpy as np

from Orange.misc import DistMatrix
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxFromDistances import OWNxFromDistances


class TestOWNxFromDistances(WidgetTest):
    def setUp(self):
        self.widget: OWNxFromDistances = self.create_widget(OWNxFromDistances)

        # Put non-zero elements in the diagonal to test if the widget ignores them
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

    def test_set_weird_matrix(self):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)
        self.assertEqual(widget.eff_distances, 10)
        self.assertIsNotNone(self.get_output(widget.Outputs.network))

        self.send_signal(widget.Inputs.distances, None)
        self.assertIsNone(self.get_output(widget.Outputs.network))

        self.send_signal(widget.Inputs.distances, self.distances)
        self.assertIsNotNone(self.get_output(widget.Outputs.network))

        self.send_signal(widget.Inputs.distances, DistMatrix(np.zeros((0, 0))))
        self.assertIsNone(self.get_output(widget.Outputs.network))

        self.send_signal(widget.Inputs.distances, DistMatrix(np.array([[1]])))
        self.assertIsNotNone(self.get_output(widget.Outputs.network))

        self.send_signal(widget.Inputs.distances, None)
        self.assertIsNone(self.get_output(widget.Outputs.network))

        self.send_signal(widget.Inputs.distances,
                         DistMatrix(np.array([[0, 1],
                                              [1, 0]])))
        self.assertIsNotNone(self.get_output(widget.Outputs.network))

    @patch("orangecontrib.network.widgets.OWNxFromDistances.Histogram.set_values")
    def test_compute_histogram_symmetric(self, set_values):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)
        np.testing.assert_almost_equal(widget.edges, [1, 2, 5, 6, 10, 13, 15])
        np.testing.assert_almost_equal(widget.cumfreqs, [1, 2, 5, 6, 7, 9, 10])
        set_values.assert_called_with(widget.edges, widget.cumfreqs)

        # The matrix is symmetric, so the number of bins is below 1000 ->
        # the histogram is computed exactly
        distances = np.zeros((40, 40))
        for i in range(40):
            distances[:i, i] = distances[i, :i] = np.arange(i)
        distances = DistMatrix(distances)
        self.send_signal(widget.Inputs.distances, distances)
        np.testing.assert_almost_equal(widget.edges, np.arange(39))
        set_values.assert_called_with(widget.edges, widget.cumfreqs)

        # Even though the matrix is symmetric, the number of bins is above 1000
        distances = np.zeros((50, 50))
        for i in range(50):
            distances[:i, i] = distances[i, :i] = np.arange(i)
        distances = DistMatrix(distances)
        self.send_signal(widget.Inputs.distances, distances)
        np.testing.assert_almost_equal(
            widget.edges[:5], [0.   , 0.048, 0.096, 0.144, 0.192])
        set_values.assert_called_with(widget.edges, widget.cumfreqs)

    @patch("orangecontrib.network.widgets.OWNxFromDistances.Histogram.set_values")
    def test_compute_histogram_asymmetric(self, set_values):
        widget = self.widget

        self.distances[0, 1] = 1.5
        self.send_signal(widget.Inputs.distances, self.distances)
        np.testing.assert_almost_equal(widget.edges, [1, 1.5, 2, 5, 6, 10, 13, 15])
        np.testing.assert_almost_equal(widget.cumfreqs, [1, 2, 4, 10, 12, 14, 18, 20])
        set_values.assert_called_with(widget.edges, widget.cumfreqs)

        distances = DistMatrix(np.array(np.arange(40 * 40).reshape((40, 40))))
        self.send_signal(widget.Inputs.distances, distances)
        np.testing.assert_almost_equal(
            widget.edges[:5], [1, 2.597, 4.194, 5.791, 7.388])
        np.testing.assert_almost_equal(
            widget.cumfreqs[:5], [2, 4, 5, 7, 8])
        self.assertEqual(widget.cumfreqs[-1], 40 * 39)
        set_values.assert_called_with(widget.edges, widget.cumfreqs)

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

    @patch("orangecontrib.network.widgets.OWNxFromDistances.OWNxFromDistances.generate_network")
    def test_set_edges(self, generate_network):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for edges, threshold, density in [(0, 0, 0), (1, 1, 10), (2, 2, 20),
                                          (3, 5, 50), (4, 5, 50), (5, 5, 50),
                                          (6, 6, 60), (7, 10, 70),
                                          (8, 13, 90), (9, 13, 90), (10, 15, 100)]:
            generate_network.reset_mock()
            self.set_edit(widget.edges_edit, edges)
            generate_network.assert_called_once()
            self.assertEqual(widget.threshold, threshold,
                             msg=f"at edges={edges}")
            self.assertEqual(widget.density, density,
                             msg=f"at edges={edges}")

    @patch("orangecontrib.network.widgets.OWNxFromDistances.OWNxFromDistances.generate_network")
    def test_set_density(self, generate_network):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for density, threshold, edges in [(0, 0, 0), (10, 1, 1), (20, 2, 2),
                                          (30, 5, 5), (40, 5, 5), (50, 5, 5),
                                          (60, 6, 6), (70, 10, 7),
                                          (80, 13, 9), (90, 13, 9), (100, 15, 10)]:
            generate_network.reset_mock()
            self.set_edit(widget.density_edit, density)
            generate_network.assert_called_once()
            self.assertEqual(widget.threshold, threshold,
                             msg=f"at density={density}")
            self.assertEqual(int(widget.edges_edit.text()), edges,
                             msg=f"at density={density}")

    @patch("orangecontrib.network.widgets.OWNxFromDistances.OWNxFromDistances.generate_network")
    def test_set_threshold(self, generate_network):
        widget = self.widget
        self.send_signal(widget.Inputs.distances, self.distances)

        for threshold, density, edges in [(0, 0, 0), (1, 10, 1), (2, 20, 2),
                                          (3, 20, 2), (4, 20, 2), (5, 50, 5),
                                          (6, 60, 6), (9, 60, 6), (10, 70, 7),
                                          (12.9, 70, 7), (13, 90, 9),
                                          (14.9, 90, 9), (15, 100, 10),
                                          (16, 100, 10)]:
            generate_network.reset_mock()
            self.set_edit(widget.threshold_edit, threshold)
            generate_network.assert_called_once()
            self.assertEqual(widget.density, density,
                             msg=f"at threshold={threshold}")
            self.assertEqual(int(widget.edges_edit.text()), edges,
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
        self.assertEqual(int(widget.edges_edit.text()), 5)

        widget.histogram.thresholdChanged.emit(11)
        generate_network.assert_not_called()
        self.assertEqual(widget.threshold, 11)
        self.assertEqual(widget.density, 70)
        self.assertEqual(int(widget.edges_edit.text()), 7)

        widget.histogram.thresholdChanged.emit(15)
        generate_network.assert_not_called()
        self.assertEqual(widget.threshold, 15)
        self.assertEqual(widget.density, 100)
        self.assertEqual(int(widget.edges_edit.text()), 10)

        widget.histogram.thresholdChanged.emit(16)
        generate_network.assert_not_called()
        self.assertEqual(widget.threshold, 16)
        self.assertEqual(widget.density, 100)
        self.assertEqual(int(widget.edges_edit.text()), 10)

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
        np.testing.assert_almost_equal(coo.data, [1, 0.01])
        self.assertEqual(list(graph.nodes), list("12345"))

        self.set_edit(widget.threshold_edit, 3)
        graph = self.get_output(widget.Outputs.network)
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 3)
        coo = graph.edges[0].edges.tocoo()
        np.testing.assert_equal(coo.row, [1, 2, 4])
        np.testing.assert_equal(coo.col, [0, 0, 1])
        np.testing.assert_almost_equal(coo.data, [1, 0.1323913, 0.01])

        self.set_edit(widget.threshold_edit, 4)
        graph = self.get_output(widget.Outputs.network)
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 3)
        coo = graph.edges[0].edges.tocoo()
        np.testing.assert_equal(coo.row, [1, 2, 4])
        np.testing.assert_equal(coo.col, [0, 0, 1])
        np.testing.assert_almost_equal(coo.data, [1, 0.1323913, 0.01])

        self.set_edit(widget.threshold_edit, 5)
        graph = self.get_output(widget.Outputs.network)
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(graph.number_of_edges(), 6)
        coo = graph.edges[0].edges.tocoo()
        np.testing.assert_equal(coo.row, [1, 2, 2, 3, 3, 4])
        np.testing.assert_equal(coo.col, [0, 0, 1, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
