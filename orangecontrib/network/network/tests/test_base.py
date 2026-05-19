import unittest

import scipy.sparse as sp
from numpy.testing import assert_equal

from orangecontrib.network.network.base import (
    Edges, DirectedEdges, UndirectedEdges)


class TestEdges(unittest.TestCase):
    def test_edges(self):
        edges = sp.csr_matrix([[0, 3, 5], [1, 0, 0], [2, 0, 3]])

        assert_equal(Edges._compose_neighbours(0, edges, False), [1, 2])
        assert_equal(Edges._compose_neighbours(1, edges, False), [0])
        assert_equal(Edges._compose_neighbours(2, edges, False), [0, 2])
        assert_equal(Edges._compose_neighbours(0, edges, True), ([1, 2], [3, 5]))
        assert_equal(Edges._compose_neighbours(1, edges, True), ([0], [1]))
        assert_equal(Edges._compose_neighbours(2, edges, True), ([0, 2], [2, 3]))

        assert_equal(Edges._compute_degrees(edges, False), [2, 1, 2])
        assert_equal(Edges._compute_degrees(edges, True), [8, 1, 5])


    def test_directed_edges(self):
        edges = sp.csr_matrix([[0, 1, 2], [5, 3, 1], [0, 7, 0]])
        edges = DirectedEdges(edges)

        assert_equal(edges.out_degrees(), [2, 3, 1])
        assert_equal(edges.in_degrees(), [1, 3, 2])
        assert_equal(edges.degrees(), [3, 6, 3])
        assert_equal(edges.out_degrees(weighted=True), [3, 9, 7])
        assert_equal(edges.in_degrees(weighted=True), [5, 11, 3])
        assert_equal(edges.degrees(weighted=True), [8, 20, 10])

        self.assertEqual(edges.out_degree(0), 2)
        self.assertEqual(edges.out_degree(1), 3)
        self.assertEqual(edges.in_degree(0), 1)
        self.assertEqual(edges.in_degree(1), 3)
        self.assertEqual(edges.out_degree(0, weighted=True), 3)
        self.assertEqual(edges.out_degree(1, weighted=True), 9)
        self.assertEqual(edges.in_degree(0, weighted=True), 5)
        self.assertEqual(edges.in_degree(1, weighted=True), 11)
        self.assertEqual(edges.degree(0), 3)
        self.assertEqual(edges.degree(1), 6)
        self.assertEqual(edges.degree(0, weighted=True), 8)
        self.assertEqual(edges.degree(1, weighted=True), 20)

        assert_equal(edges.outgoing(0), [1, 2])
        assert_equal(edges.outgoing(1), [0, 1, 2])
        assert_equal(edges.incoming(0), [1])
        assert_equal(edges.incoming(1), [0, 1, 2])
        assert_equal(edges.outgoing(0, weights=True), ([1, 2], [1, 2]))
        assert_equal(edges.outgoing(1, weights=True), ([0, 1, 2], [5, 3, 1]))
        assert_equal(edges.incoming(0, weights=True), ([1], [5]))
        assert_equal(edges.incoming(1, weights=True), ([0, 1, 2], [1, 3, 7]))

        assert_equal(edges.neighbours(0), [1, 2])
        assert_equal(edges.neighbours(1), [0, 1, 2])
        assert_equal(edges.neighbours(0, weights=True), ([1, 2], [6, 2]))
        assert_equal(edges.neighbours(1, weights=True), ([0, 1, 2], [6, 6, 8]))

    def test_undirected_edges(self):
        edges = sp.csr_matrix([[0, 0, 2], [0, 3, 1], [3, 4, 0]])
        edges = UndirectedEdges(edges)

        assert_equal(edges.degrees(), [1, 2, 2])
        assert_equal(edges.degrees(weighted=True), [5, 11, 10])

        self.assertEqual(edges.degree(0), 1)
        self.assertEqual(edges.degree(1), 2)
        self.assertEqual(edges.degree(0, weighted=True), 5)
        self.assertEqual(edges.degree(1, weighted=True), 11)

        assert_equal(edges.neighbours(0), [2])
        assert_equal(edges.neighbours(1), [1, 2])
        assert_equal(edges.neighbours(0, weights=True), ([2], [5]))
        assert_equal(edges.neighbours(1, weights=True), ([1, 2], [6, 5]))

        # These are not tests but assertions that, if true, allow us to omit
        # the redundant methods in UndirectedEdges.
        # If these assertions fail, add tests. :)
        assert UndirectedEdges.in_degrees is UndirectedEdges.out_degrees is UndirectedEdges.degrees
        assert UndirectedEdges.in_degree is UndirectedEdges.out_degree is UndirectedEdges.degree
        assert UndirectedEdges.incoming is UndirectedEdges.outgoing is UndirectedEdges.neighbours


if __name__ == "__main__":
    unittest.main()