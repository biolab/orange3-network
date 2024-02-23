import unittest

import numpy as np

from Orange.misc import DistMatrix
from Orange.data import Table, Domain, ContinuousVariable
from orangecontrib.network.widgets.utils import items_from_distmatrix


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


if __name__ == "__main__":
    unittest.main()
