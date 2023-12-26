import unittest
from unittest.mock import patch

import numpy as np

from Orange.misc import DistMatrix
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.network.widgets.OWNxNeighbor import OWNxNeighbor


class TestOWNxFromDistances(WidgetTest):
    def setUp(self):
        self.widget: OWNxNeighbor = self.create_widget(OWNxNeighbor)

        # Put non-zero elements in the diagonal to test if the widget ignores them
        self.distances = DistMatrix(np.array([[0., 1, 2, 5, 10],
                                              [1, -1, 5, 1, 13],
                                              [2, 5, 2, 6, 11],
                                              [5, 1, 6, 3, 15],
                                              [10, 13, 11, 15, 0]]))

    def test_get_neighbors(self):
        widget = self.widget
        widget.matrix = self.distances

        widget.k = 2
        nearest2 =  [(0, 1), (0, 2),
                     (1, 0), (1, 3),
                     (2, 0), (2, 1),
                     (3, 1), (3, 0),
                     (4, 0), (4, 2)]

        widget.directed = False
        edges = widget.get_neighbors()
        # when sorted, col will be lower, so zip it that way
        pairs = sorted(zip(edges.col, edges.row))
        np.testing.assert_equal(pairs, sorted({tuple(sorted(x)) for x in nearest2}))

        widget.directed = True
        edges = widget.get_neighbors()
        # zip row first; this is how nearest2 is ordered, too
        pairs = sorted(zip(edges.row, edges.col))
        np.testing.assert_equal(pairs, sorted(nearest2))

        widget.k = 1
        widget.directed = False
        edges = widget.get_neighbors()
        pairs = sorted(zip(edges.col, edges.row))
        np.testing.assert_equal(pairs, sorted({tuple(sorted(x)) for x in nearest2[::2]}))

        widget.directed = True
        edges = widget.get_neighbors()
        pairs = sorted(zip(edges.row, edges.col))
        np.testing.assert_equal(pairs, sorted(nearest2[::2]))

        nearest3 = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0),
                    (2, 1), (2, 3),
                    (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]

        widget.k = 3
        widget.directed = True
        edges = widget.get_neighbors()
        pairs = sorted(zip(edges.row, edges.col))
        np.testing.assert_equal(pairs, nearest3)

        widget.directed = False
        edges = widget.get_neighbors()
        pairs = sorted(zip(edges.col, edges.row))
        np.testing.assert_equal(pairs,
                                sorted({tuple(sorted(x)) for x in nearest3}))

        widget.k = 4
        widget.directed = True
        self.assertEqual(widget.get_neighbors().nnz, 20)

        widget.directed = False
        self.assertEqual(widget.get_neighbors().nnz, 10)

        widget.k = 6
        widget.directed = True
        self.assertEqual(widget.get_neighbors().nnz, 20)

        widget.directed = False
        self.assertEqual(widget.get_neighbors().nnz, 10)

    def test_output(self):
        nearest2 =  [(0, 1), (0, 2),
                     (1, 0), (1, 3),
                     (2, 0), (2, 1),
                     (3, 1), (3, 0),
                     (4, 0), (4, 2)]

        widget = self.widget
        widget.auto_apply = True
        widget.k = 2
        widget.directed = False
        self.send_signal(self.distances)
        widget.send_report()

        output = self.get_output()
        edges = output.edges[0].edges.tocoo()
        pairs = sorted(zip(edges.col, edges.row))
        np.testing.assert_equal(pairs, sorted({tuple(sorted(x)) for x in nearest2}))

        widget.controls.directed.click()
        output = self.get_output()
        edges = output.edges[0].edges.tocoo()
        pairs = sorted(zip(edges.row, edges.col))
        np.testing.assert_equal(pairs, sorted(nearest2))

    def test_bad_matrix(self):
        widget = self.widget
        widget.auto_apply = True

        self.send_signal(self.distances)
        self.assertIsNotNone(self.get_output())

        self.send_signal(None)
        self.assertIsNone(self.get_output())
        widget.send_report()

        self.send_signal(DistMatrix(np.zeros((0, 0))))
        self.assertIsNone(self.get_output())
        widget.send_report()

        self.send_signal(self.distances)
        self.assertIsNotNone(self.get_output())

        self.send_signal(DistMatrix(np.zeros((0, 0))))
        self.assertIsNone(self.get_output())

        self.send_signal(None)
        self.assertIsNone(self.get_output())

        self.send_signal(DistMatrix([[1]]))
        self.assertIsNone(self.get_output())
        widget.send_report()

        self.send_signal(self.distances)
        self.assertIsNotNone(self.get_output())

        self.send_signal(DistMatrix([[1]]))
        self.assertIsNone(self.get_output())


if __name__ == "__main__":
    unittest.main()