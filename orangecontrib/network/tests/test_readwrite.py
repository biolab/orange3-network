import os
import unittest

from orangecontrib.network.network import readwrite

cwd = os.path.split(__file__)[0]


class TestReadPajek(unittest.TestCase):
    def test_two_mode(self):
        davis = readwrite.read_pajek(os.path.join(cwd, "../networks/davis.net"))
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


if __name__ == "__main__":
    unittest.main()
