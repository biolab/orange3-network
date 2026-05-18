import unittest

import numpy as np

from Orange.data import Domain, Table, ContinuousVariable, StringVariable
from orangecontrib.network.network.base import DirectedEdges, UndirectedEdges
from orangecontrib.network.network.compose import (
    UnknownNodes, MismatchingEdgeVariables, NonUniqueLabels,
    network_from_tables, network_from_edge_table,
    _net_from_data_and_edges, _sort_edges, _reduced_edge_data, _edge_columns,
    _float_to_ind, _str_to_ind
)


class TestComposeBase(unittest.TestCase):
    def setUp(self) -> None:
        self.lab, self.labx = StringVariable("lab"), ContinuousVariable("x")
        self.data = Table.from_list(Domain([self.labx], None, [self.lab]),
                               [[1.3, "foo"],
                                [1.8, "bar"],
                                [2.7, "qux"],
                                [1.1, "baz"],
                                [np.nan, "bax"],
                                ])

        self.srcs, self.dsts = metas = \
            StringVariable("srcs"), StringVariable("dsts")
        self.w, self.src1, self.dst1 = attrs = [
            ContinuousVariable(x) for x in ("w src1 dst1").split()]
        self.edges = Table.from_list(Domain(attrs, None, metas),
                                [[1, 1, 4, "foo", "baz"],
                                 [8, 4, 4, "baz", "baz"],
                                 [3, 5, 2, "bax", "bar"],
                                 [2, 1, 2, "foo", "bar"]
                                 ])


class TestUtils(TestComposeBase):
    def test_net_from_data_and_edges(self):
        row_ind = self.edges.get_column(1).astype(int) - 1
        col_ind = self.edges.get_column(2).astype(int) - 1
        exp = np.zeros((5, 5), dtype=int)
        exp[row_ind, col_ind] = 1

        edge_data = self.edges.transform(Domain([self.edges.domain[0]], None))
        net = _net_from_data_and_edges(self.data, edge_data, row_ind, col_ind)
        self.assertIs(net.nodes, self.data)
        self.assertIsInstance(net.edges[0], UndirectedEdges)
        np.testing.assert_equal(net.edges[0].edges.todense(), exp)
        np.testing.assert_equal(net.edges[0].edge_data, np.array([[2, 1, 8, 3]]).T)

        net = _net_from_data_and_edges(self.data, edge_data, row_ind, col_ind,
                                       directed=True)
        self.assertIsInstance(net.edges[0], DirectedEdges)

        net = _net_from_data_and_edges(self.data, None, row_ind, col_ind)
        self.assertIs(net.nodes, self.data)
        self.assertIsInstance(net.edges[0], UndirectedEdges)
        np.testing.assert_equal(net.edges[0].edges.todense(), exp)

        net = _net_from_data_and_edges(self.data, None, row_ind, col_ind,
                                       directed=True)
        self.assertIs(net.nodes, self.data)
        self.assertIsInstance(net.edges[0], DirectedEdges)
        np.testing.assert_equal(net.edges[0].edges.todense(), exp)

    def test_sort_edges(self):
        edge_data = np.array([3, 1, 2, 7, 9, 5, 8, 6, 4])
        row_indss = np.array([2, 1, 3, 1, 2, 3, 3, 2, 1])
        col_indss = np.array([2, 2, 1, 1, 1, 2, 3, 3, 3])
        np.testing.assert_equal(_sort_edges(row_indss, col_indss, edge_data),
                                [7, 1, 4, 9, 3, 6, 2, 5, 8])

    def test_reduced_edge_data(self):
        attrs = tuple(ContinuousVariable(x) for x in "abcdefghi")
        data = Table.from_list(Domain(attrs[:3], attrs[3], attrs[4:]),
                               [[0] * 9])

        domain = _reduced_edge_data(data, attrs[0], attrs[5]).domain
        self.assertEqual(domain.attributes, attrs[1:3])
        self.assertEqual(domain.class_var, attrs[3])
        self.assertEqual(domain.metas, attrs[4:5] + attrs[6:])

        domain = _reduced_edge_data(data, attrs[2], attrs[3]).domain
        self.assertEqual(domain.attributes, attrs[:2])
        self.assertIsNone(domain.class_var)
        self.assertEqual(domain.metas, attrs[4:])

        data = Table.from_list(Domain([attrs[0]], None, [attrs[1]]), [[0, 0]])
        self.assertIsNone(_reduced_edge_data(data, attrs[0], attrs[1]))

    def test_edge_columns(self):
        c, d = ContinuousVariable("c"), ContinuousVariable("d")
        s, t = StringVariable("s"), StringVariable("t")
        domain = Domain([c], None, [d, s, t])
        edges = Table.from_list(domain, [[3, 1, "foo", "bar"],
                                         [0, 2, "bar", "baz"]])

        col1, col2 = _edge_columns(edges, c, d)
        np.testing.assert_equal(col1, [3, 0])
        np.testing.assert_equal(col2, [1, 2])

        col1, col2 = _edge_columns(edges, s, t)
        np.testing.assert_equal(col1, ["foo", "bar"])
        np.testing.assert_equal(col2, ["bar", "baz"])

        self.assertRaises(MismatchingEdgeVariables,
                          _edge_columns, edges, c, s)

    def test_str_to_ind(self):
        np.testing.assert_equal(
            _str_to_ind(np.array("foo foo baz bar".split()),
                        {"foo": 0, "bar": 1, "baz": 2}),
            [0, 0, 2, 1])

    def test_str_to_ind_errors(self):
        self.assertRaisesRegex(UnknownNodes, ".*known.*baz.*",
                               _str_to_ind,
                               np.array("foo foo baz bar".split()),
                               {"foo": 0, "bar": 1})

    def test_float_to_ind(self):
        col = np.array([1, 7, 3, 2, 4], dtype=float)
        a = _float_to_ind(col, "x")
        np.testing.assert_equal(a, col - 1)
        self.assertEqual(a.dtype, int)

        a = _float_to_ind(col, "x", 7)
        np.testing.assert_equal(a, col - 1)
        self.assertEqual(a.dtype, int)

    def test_float_to_ind_errors(self):
        self.assertRaisesRegex(
            UnknownNodes, ".*missing values.*",
            _float_to_ind, np.array([1, 7, 3, np.nan, 4]), "x"
        )
        self.assertRaisesRegex(
            UnknownNodes, ".*non-integer.*",
            _float_to_ind, np.array([1, 7, 3, 1.5, 4]), "x"
        )
        self.assertRaisesRegex(
            UnknownNodes, ".*negative.*",
            _float_to_ind, np.array([1, 7, 3, -1, 4]), "x"
        )
        self.assertRaisesRegex(
            UnknownNodes, ".*1-based.*",
            _float_to_ind, np.array([1, 7, 3, 0, 4]), "x"
        )
        self.assertRaisesRegex(
            UnknownNodes, ".*large.*",
            _float_to_ind, np.array([1, 7, 3, 2, 4]), "x", 5
        )
        self.assertRaisesRegex(
            UnknownNodes, ".*large.*",
            _float_to_ind, np.array([1, 7, 3e10, 2, 4]), "x", 5
        )


class TestFunctions(TestComposeBase):
    def test_network_from_tables(self):
        exp_edges = np.zeros((5, 5), dtype=int)
        exp_edges[0, 1] = exp_edges[0, 3] = exp_edges[3, 3] = exp_edges[4, 1] = 1
        odom = self.edges.domain

        for src, dst, tst in ((self.srcs, self.dsts, "edges as strings"),
                              (self.src1, self.dst1, "edges as indices")):
            with self.subTest(tst):
                network = network_from_tables(self.data, self.lab,
                                              self.edges, src, dst)
                self.assertIsInstance(network.edges[0], UndirectedEdges)
                np.testing.assert_equal(network.edges[0].edges.todense(), exp_edges)
                self.assertIs(network.nodes, self.data)

                network = network_from_tables(self.data, self.lab,
                                              self.edges, src, dst,
                                              directed=True)
                self.assertIsInstance(network.edges[0], DirectedEdges)
                np.testing.assert_equal(network.edges[0].edges.todense(), exp_edges)
                # node data is assigned as is
                self.assertIs(network.nodes, self.data)
                # edge data must be sorted by rows, then columns
                np.testing.assert_equal(network.edges[0].edge_data[:, 0],
                                        np.array([[2, 1, 8, 3]]).T)
                # edge data must not contain the used columns
                edom = network.edges[0].edge_data.domain
                self.assertEqual(len(edom.attributes + edom.metas),
                                 len(odom.attributes + odom.metas) - 2)

        # just two attributes in edges -> no edge_data afterwards
        red_edges = self.edges.transform(Domain([], None, odom.metas))
        network = network_from_tables(self.data, self.lab,
                                      red_edges, self.srcs, self.dsts)
        self.assertIsInstance(network.edges[0], UndirectedEdges)
        np.testing.assert_equal(network.edges[0].edges.todense(), exp_edges)
        self.assertIs(network.nodes, self.data)
        self.assertIsNone(network.edges[0].edge_data)

    def test_network_from_tables_errors(self):
        self.assertRaises(
            MismatchingEdgeVariables,
            network_from_tables,
            self.data, self.lab,
            self.edges, self.srcs, self.dst1
        )

        with self.data.unlocked(self.data.metas):
            self.data.metas[2, 0] = self.data.metas[0, 0]

        self.assertRaises(
            NonUniqueLabels,
            network_from_tables,
            self.data, self.lab,
            self.edges, self.srcs, self.dst1
        )

    def test_network_from_edge_table_by_labels(self):
        network = network_from_edge_table(self.edges, self.srcs, self.dsts)
        odom = self.edges.domain

        # labels are assumed to be sorted alphabetically
        labels = {lab: i for i, lab in enumerate("bar bax baz foo".split())}
        exp_edges = np.zeros((4, 4), dtype=int)
        for src, dst in [("foo", "baz"), ("baz", "baz"),
                         ("bax", "bar"), ("foo", "bar")]:
            exp_edges[labels[src], labels[dst]] = 1

        self.assertIsInstance(network.edges[0], UndirectedEdges)
        np.testing.assert_equal(network.edges[0].edges.todense(), exp_edges)

        network = network_from_edge_table(self.edges, self.srcs, self.dsts,
                                          directed=True)
        self.assertIsInstance(network.edges[0], DirectedEdges)
        np.testing.assert_equal(network.edges[0].edges.todense(), exp_edges)

        # edge data contains only labels
        np.testing.assert_equal(network.nodes.metas.T, [list(labels)])

        # edge data must be sorted labels
        np.testing.assert_equal(network.edges[0].edge_data[:, 0],
                                np.array([[3, 8, 2, 1]]).T)
        # edge data must not contain the used columns
        edom = network.edges[0].edge_data.domain
        self.assertEqual(len(edom.attributes + edom.metas),
                         len(odom.attributes + odom.metas) - 2)

    def test_network_from_edge_table_by_indices(self):
        exp_edges = np.zeros((5, 5), dtype=int)
        exp_edges[0, 1] = exp_edges[0, 3] = exp_edges[3, 3] = exp_edges[4, 1] = 1
        odom = self.edges.domain

        network = network_from_edge_table(self.edges, self.src1, self.dst1)
        self.assertIsInstance(network.edges[0], UndirectedEdges)
        np.testing.assert_equal(network.edges[0].edges.todense(), exp_edges)

        network = network_from_edge_table(self.edges, self.src1, self.dst1,
                                          directed=True)
        self.assertIsInstance(network.edges[0], DirectedEdges)
        np.testing.assert_equal(network.edges[0].edges.todense(), exp_edges)

        # edge data contains only numbers
        np.testing.assert_equal(network.nodes.metas.T, [list("12345")])

        # edge data must be sorted by rows, then columns
        np.testing.assert_equal(network.edges[0].edge_data[:, 0],
                                np.array([[2, 1, 8, 3]]).T)
        # edge data must not contain the used columns
        edom = network.edges[0].edge_data.domain
        self.assertEqual(len(edom.attributes + edom.metas),
                         len(odom.attributes + odom.metas) - 2)


if __name__ == '__main__':
    unittest.main()
