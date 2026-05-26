import os
import unittest
from importlib.resources import files
from tempfile import NamedTemporaryFile

import numpy as np

from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable
from orangecontrib.network.network import readwrite
from orangecontrib.network.network.readwrite import dict_rows_from_table, \
    table_from_dicts


def _fullpath(name):
    return str(files("orangecontrib.network").joinpath("networks", name))


def _fullpathtest(name):
    # Not a package, so we can't use importlib.resources
    return os.path.join(os.path.dirname(__file__), name)


class TestReadPajek(unittest.TestCase):
    def test_two_mode(self):
        davis = readwrite.read_pajek(_fullpath("davis.net"))
        self.assertEqual(davis.number_of_nodes(), 32)
        self.assertEqual(
            list(davis.nodes),
            ['EVELYN', 'LAURA', 'THERESA', 'BRENDA', 'CHARLOTTE', 'FRANCES',
             'ELEANOR', 'PEARL', 'RUTH', 'VERNE', 'MYRNA', 'KATHERINE',
             'SYLVIA', 'NORA', 'HELEN', 'DOROTHY', 'OLIVIA', 'FLORA', 'E1',
             'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11',
             'E12', 'E13', 'E14']
            )
        self.assertEqual(davis.in_first_mode, 18)

    def test_edge_labels(self):
        net = readwrite.read_pajek(_fullpathtest("towns.net"))
        self.assertEqual(net.number_of_nodes(), 4)
        self.assertEqual(
            list(net.nodes),
            ["Ljubljana", "Kranj", "Maribor", "Novo mesto"]
            )
        self.assertEqual(
            list(net.edges[0].edge_data),
            ['near', 'far', 'not near, not far', 'huh?'])

    def test_data(self):
        net = readwrite.read_pajek(_fullpathtest("towns-2.net"))
        self.do_test(net)

    def do_test(self, net):
        self.assertEqual(net.number_of_nodes(), 5)
        self.assertIsInstance(net.nodes.domain["population"], ContinuousVariable)
        self.assertIsInstance(net.nodes.domain["capital"], DiscreteVariable)
        self.assertIsInstance(net.nodes.domain["region"], StringVariable)
        self.assertIsInstance(net.nodes.domain["node label"], StringVariable)
        np.testing.assert_equal(
            net.nodes.get_column("node label"),
            ["Ljubljana", "Kranj", "Maribor", "Novo mesto", "Celje"])
        np.testing.assert_equal(
            net.nodes.get_column("population"),
            [300000, 38000, 110000, np.nan, np.nan])
        np.testing.assert_equal(
            net.nodes.get_column("capital"),
            [1, np.nan, np.nan, 0, np.nan])
        np.testing.assert_equal(
            net.nodes.get_column("region"),
            ["central", "gorenjska", "štajerska", "dolenjska", ""])

        data = net.edges[0].edge_data
        self.assertIsInstance(data.domain["distance"], ContinuousVariable)
        self.assertIsInstance(data.domain["qdist"], StringVariable)
        self.assertIsInstance(data.domain["relations"], DiscreteVariable)

        np.testing.assert_equal(
            data.get_column("distance"), [30, 130, 74, np.nan])
        np.testing.assert_equal(
            data.get_column("qdist"), ["near", "far", "not near, not far", ""])
        np.testing.assert_equal(
            data.get_column("relations"), [np.nan, 0, np.nan, np.nan])

    def test_write_data(self):
        net = readwrite.read_pajek(_fullpathtest("towns-2.net"))
        with NamedTemporaryFile("wt", suffix=".net", delete=False, encoding="utf-8") as f:
            try:
                readwrite.write_pajek(
                    f, net, net.nodes.get_column("node label"),
                    dict_rows_from_table(net.nodes, exclude="node label"),
                    dict_rows_from_table(net.edges[0].edge_data, exclude="edge label"))
                f.close()
                net2 = readwrite.read_pajek(f.name)
                self.do_test(net2)
            finally:
                os.remove(f.name)

    def test_write_pajek(self):
        net = readwrite.read_pajek(_fullpath("../networks/leu_by_genesets.net"))
        with NamedTemporaryFile("wt", suffix=".net", delete=False) as f:
            try:
                readwrite.write_pajek(f, net)
                f.close()
                net2 = readwrite.read_pajek(f.name)
                np.testing.assert_equal(net2.nodes, net.nodes)
                np.testing.assert_equal(net2.coordinates, net.coordinates)
                self.assertEqual(len(net2.edges), 1)
                edges, edges2 = net.edges[0].edges, net2.edges[0].edges
                np.testing.assert_equal(edges.indptr, edges2.indptr)
                np.testing.assert_equal(edges.indices, edges2.indices)
                np.testing.assert_almost_equal(edges.data, edges2.data)
            finally:
                os.remove(f.name)

    def test_write_pajek_no_coordinates(self):
        net = readwrite.read_pajek(_fullpath("../networks/leu_by_genesets.net"))
        net.coordinates = None
        with NamedTemporaryFile("wt", suffix=".net", delete=False) as f:
            try:
                readwrite.write_pajek(f, net)
                f.close()
                net2 = readwrite.read_pajek(f.name)
                np.testing.assert_equal(net2.nodes, net.nodes)
                self.assertIsNone(net2.coordinates, net.coordinates)
            finally:
                os.remove(f.name)

    def test_write_pajek_multiple_edge_types(self):
        net = readwrite.read_pajek(_fullpath("../networks/leu_by_genesets.net"))
        net.edges.append(net.edges[0])
        with NamedTemporaryFile("wt", suffix=".net") as f:
            self.assertRaises(TypeError, readwrite.write_pajek, f, net)

    def test_edge_list(self):
        net = readwrite.read_pajek(_fullpathtest("test-arcslist.net"))
        neighs = [(1, (2, 3, 6)),
                  (2, (1, 4, 5, 6)),
                  (5, (1, 2)),
                  (6, (2, 3, 4))]
        self.assertEqual(net.number_of_edges(), sum(len(y) for _, y in neighs))
        self.assertTrue(net.edges[0].directed)
        for x, y in neighs:
            np.testing.assert_equal(net.outgoing(x - 1), np.array(y) - 1)


class TestUtils(unittest.TestCase):
    def test_is_number(self):
        self.assertTrue(readwrite.is_number("1"))
        self.assertTrue(readwrite.is_number("1.5"))
        self.assertTrue(readwrite.is_number("-1.5"))
        self.assertFalse(readwrite.is_number("abc"))
        self.assertFalse(readwrite.is_number("1,5"))
        self.assertFalse(readwrite.is_number("nan"))

    def test_dicts_from_table(self):
        from Orange.data import Table, Domain
        domain = Domain([ContinuousVariable("a a"),
                         DiscreteVariable("b", ("red", "light green"))],
                        metas=[StringVariable("c\"")])
        table = Table.from_list(domain, [[1, 0, ""],
                                         [2, np.nan, "y"],
                                         [np.nan, 1, "z"],
                                         [np.nan, np.nan, ""]])
        rows = dict_rows_from_table(table)
        self.assertEqual(
            rows,
            ['"a a" 1 b red', '"a a" 2 "c\\"" y', 'b "light green" "c\\"" z', ""])

    def test_table_from_dicts(self):
        dicts = [{"a": 1, "b": "red", "c": "x"},
                 {"a": 2, "c": "y"},
                 {"b": "green", "c": "z"},
                 {}]
        table = table_from_dicts(dicts)
        np.testing.assert_equal(
            table.X,
            [[1, 1],
             [2, np.nan],
             [np.nan, 0],
             [np.nan, np.nan]])
        np.testing.assert_equal(table.metas, [["x"], ["y"], ["z"], [""]])

        table = table_from_dicts(
            dicts, label_name="a", label_values=["foo", "bar", "baz", "qux"])
        np.testing.assert_equal(
            table.X,
            [[1, 1],
             [2, np.nan],
             [np.nan, 0],
             [np.nan, np.nan]])
        np.testing.assert_equal(
            table.metas,
            [["x", "foo"], ["y", "bar"], ["z", "baz"], ["", "qux"]])
        self.assertEqual(table.domain.metas[1].name, "a (1)")


if __name__ == "__main__":
    unittest.main()
