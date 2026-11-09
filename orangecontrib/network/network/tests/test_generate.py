import unittest
from itertools import combinations

import numpy as np

from orangecontrib.network import Network
from orangecontrib.network.network import generate


def edge_set(net):
    """Set of undirected edges (as sorted pairs) of the network"""
    coo = net.edges[0].edges.tocoo()
    return {tuple(sorted(e)) for e in zip(coo.row.tolist(), coo.col.tolist())}


class TestGenerate(unittest.TestCase):
    def assert_network(self, net, n_nodes, edges, name=None):
        self.assertIsInstance(net, Network)
        self.assertEqual(net.number_of_nodes(), n_nodes)
        self.assertEqual(edge_set(net), set(edges))
        self.assertEqual(net.edges[0].edges.nnz, len(edges))
        if name is not None:
            self.assertEqual(net.name, name)

    def test_path(self):
        self.assert_network(
            generate.path(4), 4, [(0, 1), (1, 2), (2, 3)], "path(4)")
        self.assert_network(generate.path(1), 1, [])

    def test_cycle(self):
        self.assert_network(
            generate.cycle(4), 4, [(0, 1), (1, 2), (2, 3), (0, 3)],
            "cycle(4)")
        self.assert_network(
            generate.cycle(3), 3, [(0, 1), (1, 2), (0, 2)])

    def test_complete(self):
        for n in (1, 2, 5):
            self.assert_network(
                generate.complete(n), n, list(combinations(range(n), 2)),
                f"complete({n})")

    def test_complete_bipartite(self):
        self.assert_network(
            generate.complete_bipartite(2, 3), 5,
            [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)],
            "complete_bipartite(2, 3)")
        self.assert_network(
            generate.complete_bipartite(3, 1), 4, [(0, 3), (1, 3), (2, 3)])

    def test_barbell(self):
        self.assert_network(
            generate.barbell(3, 2), 5,
            [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)], "barbell(3, 2)")
        self.assert_network(
            generate.barbell(2, 3), 5,
            [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4)])

    def test_barbell_edge_count(self):
        n, m = 5, 4
        net = generate.barbell(n, m)
        self.assertEqual(net.number_of_nodes(), n + m)
        self.assertEqual(net.edges[0].edges.nnz,
                         n * (n - 1) // 2 + m * (m - 1) // 2 + 1)

    def test_grid(self):
        self.assert_network(
            generate.grid(3, 2), 6,
            [(0, 1), (1, 2), (3, 4), (4, 5),
             (0, 3), (1, 4), (2, 5)],
            "grid(3, 2)")
        n, m = 4, 3
        net = generate.grid(n, m)
        self.assertEqual(net.number_of_nodes(), n * m)
        self.assertEqual(net.edges[0].edges.nnz,
                         (n - 1) * m + n * (m - 1))

    def test_grid_circular(self):
        # rows are cycles of length 3, columns are paths
        self.assert_network(
            generate.grid(3, 2, True), 6,
            [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
             (0, 3), (1, 4), (2, 5)],
            "grid(3, 2, True)")

    def test_grid_circular_more_rows(self):
        n, m = 4, 3
        expected = set()
        for r in range(m):
            for c in range(n):
                i = r * n + c
                expected.add(tuple(sorted((i, r * n + (c + 1) % n))))
                if r + 1 < m:
                    expected.add((i, i + n))
        self.assert_network(generate.grid(n, m, True), n * m, expected)

    def test_ladder(self):
        net = generate.ladder(3)
        self.assert_network(net, 6, edge_set(generate.grid(3, 2)))
        self.assertEqual(net.name, "grid(3, 2)")

    def test_circular_ladder(self):
        # prism graph: two cycles joined by rungs
        self.assert_network(
            generate.circular_ladder(3), 6,
            [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5),
             (0, 3), (1, 4), (2, 5)])

    def test_hypercube(self):
        net = generate.hypercube(2)
        self.assertEqual(net.name, "hypercube(2)")
        self.assertEqual(net.number_of_nodes(), 4)
        self.assertEqual(edge_set(net), {(0, 1), (0, 2), (1, 3), (2, 3)})

        net = generate.hypercube(3)
        self.assertEqual(net.number_of_nodes(), 8)
        # nodes are connected iff their binary labels differ in one bit
        expected = {(i, j) for i, j in combinations(range(8), 2)
                    if bin(i ^ j).count("1") == 1}
        self.assertEqual(edge_set(net), expected)
        np.testing.assert_equal(net.degrees(), np.full(8, 3))

    def test_hypercube_zero(self):
        net = generate.hypercube(0)
        self.assertEqual(net.number_of_nodes(), 1)
        self.assertEqual(edge_set(net), set())

    def test_hypercube_number_of_edges(self):
        # Currently both directions of each edge are stored, so the number
        # of edges is doubled, unlike in other generators
        for ndim in range(1, 5):
            self.assertEqual(generate.hypercube(ndim).number_of_edges(),
                             ndim * 2 ** (ndim - 1))

    def test_star(self):
        self.assert_network(
            generate.star(3), 4, [(0, 1), (0, 2), (0, 3)], "star(3)")
        self.assertEqual(generate.star(5).degree(0), 5)

    def test_lollipop(self):
        self.assert_network(
            generate.lollipop(3, 2), 5,
            [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)], "lollipop(3, 2)")
        self.assert_network(
            generate.lollipop(4, 2), 6,
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])

    def test_geometric(self):
        np.random.seed(42)
        net = generate.geometric(20, 30)
        self.assertEqual(net.name, "geometric(20,30)")
        self.assertEqual(net.number_of_nodes(), 20)
        self.assertEqual(net.coordinates.shape, (20, 2))
        self.assertTrue(np.all((net.coordinates >= 0) & (net.coordinates < 1)))
        # ties in distances are practically impossible with random points
        self.assertEqual(len(edge_set(net)), 30)

    def test_geometric_complete(self):
        np.random.seed(42)
        net = generate.geometric(4, 6)
        self.assertEqual(net.name, "geometric(4,6)")
        self.assertEqual(net.number_of_nodes(), 4)
        self.assertEqual(net.coordinates.shape, (4, 2))
        self.assertTrue(np.all((net.coordinates >= 0) & (net.coordinates < 1)))
        np.testing.assert_equal(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
            net.edges[0].edges.todense())

    def test_geometric_empty(self):
        net = generate.geometric(4, 0)
        self.assertEqual(net.name, "geometric(4,0)")
        self.assertEqual(net.number_of_nodes(), 4)
        self.assertEqual(net.coordinates.shape, (4, 2))
        self.assertTrue(np.all((net.coordinates >= 0) & (net.coordinates < 1)))
        np.testing.assert_equal(
            np.zeros((4, 4)), net.edges[0].edges.todense())

    def test_geometric_connects_closest_pairs(self):
        np.random.seed(0)
        n, k = 15, 20
        net = generate.geometric(n, k)
        xy = net.coordinates
        dist = {(i, j): np.sum((xy[i] - xy[j]) ** 2)
                for i, j in combinations(range(n), 2)}
        closest = set(sorted(dist, key=dist.get)[:k])
        self.assertEqual(edge_set(net), closest)

    def test_geometric_too_many_edges(self):
        self.assertRaises(ValueError, generate.geometric, 4, 7)
        self.assertEqual(generate.geometric(4, 5).number_of_nodes(), 4)

    def test_nodes_are_range(self):
        for net in (generate.path(4), generate.complete(4),
                    generate.hypercube(2), generate.star(3)):
            np.testing.assert_equal(
                np.asarray(net.nodes), np.arange(net.number_of_nodes()))


if __name__ == "__main__":
    unittest.main()
