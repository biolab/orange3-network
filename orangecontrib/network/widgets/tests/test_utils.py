import unittest

import numpy as np

from Orange.misc import DistMatrix
from Orange.data import Table, Domain, ContinuousVariable
from orangecontrib.network.widgets.utils import items_from_distmatrix, weights_from_distances


class TestItemsFromMatrix(unittest.TestCase):
    def test_items_from_matrix(self):
        distances = DistMatrix(np.array([[0., 1, 2, 5, 10],
                                         [1, -1, 5, 5, 13],
                                         [2, 5, 2, 6, 13],
                                         [5, 5, 6, 3, 15],
                                         [10, 13, 13, 15, 0]]))

        self.assertEqual(list(items_from_distmatrix(distances).metas[:, 0]),
                         ["1", "2", "3", "4", "5"])

        distances.row_items = list("abcde")
        self.assertEqual(list(items_from_distmatrix(distances).metas[:, 0]),
                         list("abcde"))

        distances.row_items = Table.from_numpy(
            Domain([ContinuousVariable(x) for x in "abcde"]),
            np.arange(25).reshape(5, 5))

        distances.axis = 1
        self.assertIs(items_from_distmatrix(distances), distances.row_items)

        distances.axis = 0
        self.assertEqual(list(items_from_distmatrix(distances).metas[:, 0]),
                         list("abcde"))

    def test_weights_from_distances(self):
        weights = np.array([])
        self.assertEqual(weights_from_distances(weights).size, 0)

        weights = np.array([1])
        np.testing.assert_almost_equal(weights_from_distances(weights), [1])

        weights = np.array([1, 1, 1, 1, 1])
        np.testing.assert_almost_equal( weights_from_distances(weights), [1, 1, 1, 1, 1])

        weights = np.array([0, 0, 0, 0, 0])
        np.testing.assert_almost_equal(weights_from_distances(weights), [1, 1, 1, 1, 1])

        weights = np.array([0, 1])
        np.testing.assert_almost_equal(weights_from_distances(weights), [1, 0.1])

        weights = np.array([2, 3, 4])
        np.testing.assert_almost_equal(weights_from_distances(weights), [1 / 10 ** 0.5, 1 / 10 ** 0.75, 0.1])

        weights = np.array([4, 3, 2])
        np.testing.assert_almost_equal(weights_from_distances(weights), [0.1, 1 / 10 ** 0.75, 1 / 10 ** 0.5])

        weights = np.array([400, 300, 200])
        np.testing.assert_almost_equal(weights_from_distances(weights), [0.1, 1 / 10 ** 0.75, 1 / 10 ** 0.5])

        weights = np.array([400, 300, 200, 0])
        np.testing.assert_almost_equal(weights_from_distances(weights), [0.1, 1 / 10 ** 0.75, 1 / 10 ** 0.5, 1])

        weights = np.array([0.00000007, 0.00000008, 0.00000009])
        np.testing.assert_almost_equal(weights_from_distances(weights), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
