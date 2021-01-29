import os
import unittest
from tempfile import NamedTemporaryFile

import numpy as np

from orangecontrib.network.network import readwrite


def _fullpath(name):
    return os.path.join(os.path.split(__file__)[0], name)


class TestReadPajek(unittest.TestCase):
    def test_two_mode(self):
        davis = readwrite.read_pajek(_fullpath("../networks/davis.net"))
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
        net = readwrite.read_pajek(_fullpath("test-arcslist.net"))
        neighs = [(1, (2, 3, 6)),
                  (2, (1, 4, 5, 6)),
                  (5, (1, 2)),
                  (6, (2, 3, 4))]
        self.assertEqual(net.number_of_edges(), sum(len(y) for _, y in neighs))
        self.assertTrue(net.edges[0].directed)
        for x, y in neighs:
            np.testing.assert_equal(net.outgoing(x - 1), np.array(y) - 1)


if __name__ == "__main__":
    unittest.main()
