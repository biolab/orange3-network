# pylint: disable=protected-access

import unittest

import numpy as np
import scipy.sparse as sp
import orangecontrib.network.network.twomode as tm
from orangecontrib.network import Network
from orangecontrib.network.widgets.tests.utils import _create_net


class TestTwoMode(unittest.TestCase):
    def test_normalization(self):
        def edges_almost_equal(a, edges):
            a = a.todense()
            a2 = np.zeros(a.shape)
            for n1, n2, w in edges:
                a2[n1, n2] = w
            np.testing.assert_almost_equal(a, a2)
            self.assertEqual(a.dtype, float)

        edges = sp.coo_matrix(
            ([1., 5, 3, 4, 2, 6], ([0, 1, 1, 2, 2, 3], [4, 4, 5, 4, 5, 6])),
            (7, 7)
        )

        s = np.sqrt
        edges_almost_equal(
            tm.Weighting[tm.NoWeights].func(edges),
            ((0, 1, 1), (0, 2, 1), (1, 2, 1))
        )
        edges_almost_equal(
            tm.Weighting[tm.WeightConnections].func(edges),
            ((0, 1, 1), (0, 2, 1), (1, 2, 2))
        )
        edges_almost_equal(
            tm.Weighting[tm.WeightWeightedConnections].func(edges),
            ((0, 1, 5), (0, 2, 4), (1, 2, 26))
        )
        edges_almost_equal(
            tm.Weighting[tm.WeightGeo].func(edges),
            ((0, 1, 1 / s(10) * 5 / s(80)),
             (0, 2, 1 / s(10) * 4 / s(60)),
             (1, 2, 5 / s(80) * 4 / s(60) + 3 / s(40) * 2 / s(30)))
        )
        edges_almost_equal(
            tm.Weighting[tm.WeightGeoDeg].func(edges),
            ((0, 1, 1 / s(3) * 5 / s(6)),
             (0, 2, 1 / s(3) * 4 / s(6)),
             (1, 2, 5 / s(6) * 4 / s(6) + 3 / s(4) * 2 / s(4)))
        )
        edges_almost_equal(
            tm.Weighting[tm.WeightInput].func(edges),
            ((0, 1, 1 / 1 * 5 / 8),
             (0, 2, 1 / 1 * 4 / 6),
             (1, 2, 5 / 8 * 4 / 6 + 3 / 8 * 2 / 6))
        )
        edges_almost_equal(
            tm.Weighting[tm.WeightOutput].func(edges),
            ((0, 1, 1 / 10 * 5 / 10),
             (0, 2, 1 / 10 * 4 / 10),
             (1, 2, 5 / 10 * 4 / 10 + 3 / 5 * 2 / 5))
        )
        edges_almost_equal(
            tm.Weighting[tm.WeightMin].func(edges),
            ((0, 1, 1 / 1 * 5 / 8),
             (0, 2, 1 / 1 * 4 / 6),
             (1, 2, 5 / 8 * 4 / 6 + 3 / 5 * 2 / 5))
        )
        edges_almost_equal(
            tm.Weighting[tm.WeightMax].func(edges),
            ((0, 1, 1 / 10 * 5 / 10),
             (0, 2, 1 / 10 * 4 / 10),
             (1, 2, 5 / 10 * 4 / 10 + 3 / 8 * 2 / 6))
        )

    def test_filtered_edges(self):
        def assert_edges(actual, expected):
            self.assertEqual(len(actual.data), len(expected))
            self.assertEqual(actual.data.dtype, float)
            self.assertEqual(
                set(zip(actual.row, actual.col, actual.data)), set(expected))

        net = _create_net(((0, 4, 1.), (4, 1, 5), (1, 5, 3),
                           (2, 4, 4), (2, 5, 2), (3, 6, 6)))

        # All edges
        assert_edges(
            tm._filtered_edges(
                net,
                np.array([True] * 4 + [False] * 3),
                np.array([False] * 4 + [True] * 3)),
            ((0, 4, 1.), (1, 4, 5.), (1, 5, 3.),
             (2, 4, 4.), (2, 5, 2.), (3, 6, 6.))
        )

        # All edges, opposite mode roles
        assert_edges(
            tm._filtered_edges(
                net,
                np.array([False] * 4 + [True] * 3),
                np.array([True] * 4 + [False] * 3)),
            ((0, 0, 1.), (0, 1, 5.), (1, 1, 3.),
             (0, 2, 4.), (1, 2, 2.), (2, 3, 6.))
        )

        # Not all edges
        assert_edges(
            tm._filtered_edges(
                net,
                np.array([True] * 4 + [False] * 3),
                np.array([False] * 5 + [True] * 2)),
            ((1, 5, 3.), (2, 5, 2.), (3, 6, 6.))
        )

        # One mode is empty
        assert_edges(
            tm._filtered_edges(
                net,
                np.array([True] * 4 + [False] * 3),
                np.array([False] * 7)),
            ()
        )

        # The other mode is empty
        assert_edges(
            tm._filtered_edges(
                net,
                np.array([False] * 7),
                np.array([False] * 5 + [True] * 2)),
            ()
        )

        # Both modes are empty
        assert_edges(
            tm._filtered_edges(
                net,
                np.array([False] * 7),
                np.array([False] * 7)),
            ()
        )

        # Graph is empty
        net = Network(range(7), sp.csr_matrix((7, 7)))
        assert_edges(
            tm._filtered_edges(
                net,
                np.array([True] * 4 + [False] * 3),
                np.array([False] * 7)),
            ()
        )

    def test_to_single_mode(self):
        net = _create_net(((0, 4, 1.), (4, 1, 5), (1, 5, 3),
                           (2, 4, 4), (2, 5, 2), (3, 6, 6)))

        net1 = tm.to_single_mode(
            net,
            np.array([True] * 4 + [False] * 3),
            np.array([False] * 4 + [True] * 3),
            tm.NoWeights)

        np.testing.assert_equal(
            net1.edges[0].edges.todense(),
            np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        )

        net = _create_net(((0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)), n=5)
        net2 = tm.to_single_mode(
            net,
            np.array([True, False, False, True, False]),
            np.array([False, False, True, False, True]),
            tm.NoWeights
        )
        np.testing.assert_equal(
            net2.edges[0].edges.todense(),
            np.array([[0, 1],
                      [0, 0]])
        )


if __name__ == "__main__":
    unittest.main()
