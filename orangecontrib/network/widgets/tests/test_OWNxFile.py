import os
import unittest
from unittest.mock import patch, Mock

import numpy as np
import scipy.sparse as sp

import Orange
from Orange.data import Table, DiscreteVariable, ContinuousVariable, \
    StringVariable, Domain
from orangecontrib.network import Network
from orangecontrib.network.network.base import DirectedEdges
from orangecontrib.network.widgets.OWNxFile import OWNxFile
from orangecontrib.network.widgets.tests.utils import NetworkTest

TEST_NETS = os.path.join(os.path.split(__file__)[0], "networks")


def _get_test_net(filename):
    return os.path.join(TEST_NETS, filename)


def select(combo, var):
    ind = combo.model().indexOf(var)
    combo.setCurrentIndex(ind)
    combo.activated[int].emit(ind)


class TestOWNxFile(NetworkTest):
    def setUp(self):
        self.widget = self.create_widget(OWNxFile)  # type: OWNxFile

        self.lab, self.lab2, self.labx = (StringVariable("lab"),
                                          StringVariable("lab2"),
                                          ContinuousVariable("x"))
        self.data = Table.from_list(Domain([self.labx],
                                           None,
                                           [self.lab, self.lab2]),
                                    [[1.3, "foo", "a"],
                                     [1.8, "bar", "b"],
                                     [2.7, "qux", "c"],
                                     [1.1, "baz", "d"],
                                     [np.nan, "bax", "e"],
                                     ])

        self.srcs, self.dsts = metas = \
            StringVariable("srcs"), StringVariable("dsts")
        self.w, self.src1, self.dst1, self.spam = attrs = [
            ContinuousVariable(x) for x in ("w src1 dst1 spam").split()]
        self.edges = Table.from_list(Domain(attrs, None, metas),
                                     [[1, 1, 4, 1.5, "foo", "baz"],
                                      [8, 4, 4, 2, "baz", "baz"],
                                      [3, 5, 2, 5, "bax", "bar"],
                                      [2, 1, 2, 0, "foo", "bar"]
                                      ])

        edges = np.zeros((5, 5), dtype=int)
        edges[0, 3] = edges[3, 3] = edges[4, 1] = edges[0, 1] = 1
        self.network = Network("foo bar qux baz bax".split(),
                               sp.csr_array(edges))

    def test_read_error(self):
        with patch("orangecontrib.network.widgets.OWNxFile.read_pajek",
                   Mock(side_effect=OSError)):
            self.widget.open_net_file("foo.net")
        self.assertTrue(self.widget.Error.io_error.is_shown())
        filename = self._get_filename("leu_by_genesets.net")
        self.widget.open_net_file(filename)
        self.assertFalse(self.widget.Error.io_error.is_shown())

    def test_load_datafile(self):
        self.widget.open_net_file(_get_test_net("test.net"))
        items = self.get_output(self.widget.Outputs.items)
        self.assertEqual(items[0]["name"], "aaa")

    def test_invalid_datafile_length(self):
        # When data file's length does not match, the widget must create
        # a table from node labels
        self.widget.open_net_file(_get_test_net("test_inv.net"))
        self.assertTrue(self.widget.Warning.mismatched_lengths)

        network = self.get_output(self.widget.Outputs.network)
        self.assertEqual(network.number_of_nodes(), 7)

        items = self.get_output(self.widget.Outputs.items)
        self.assertEqual(len(items), 7)
        self.assertEqual(items[0]["node_label"], "aa")

    def test_vars_for_label(self):
        self.widget.open_net_file(self._get_filename(None, mode="t"))

        self.widget.data = Orange.data.Table(_get_test_net("test_data.tab"))
        domain = self.widget.data.domain
        best_var, useful_vars = self.widget._vars_for_label()
        self.assertIs(best_var, domain["label"])
        self.assertEqual(useful_vars, [domain["with_extras"], domain["label"]])

        data = Orange.data.Table.from_list(
            Orange.data.Domain(
                [], None, [Orange.data.StringVariable(x) for x in "abcde"]),
            [["aa", "", "cc", "aa", ""],
             ["bb", "bb", "cc", "bb", "aa"],
             ["cc", "", "aa", "cc", "bb"],
             ["dd", "aa", "bb", "dd", "cc"],
             ["ee", "cc", "dd", "ee", ""],
             ["ff", "ee", "ee", "ff", "dd"],
             ["gg", "dd", "ff", "", "ee"],
             ["hh", "ff", "gg", "", "ff"],
             ["ii", "gg", "", "", "gg"]]
        )
        domain = data.domain
        self.widget.data = data
        best_var, useful_vars = self.widget._vars_for_label()
        self.assertIs(best_var, domain["b"])
        # c is not unique and d doesn't cover all values
        self.assertEqual(useful_vars, [domain["a"], domain["b"], domain["e"]])

    def test_label_combo_contents(self):
        widget = self.widget
        widget.read_auto_data = Mock()

        widget.open_net_file(self._get_filename(None, mode="t"))
        self.assertEqual(list(widget.label_model), [None])

        data = Orange.data.Table(_get_test_net("test_data.tab"))
        domain = data.domain

        self.send_signal(widget.Inputs.items, data)

        # Model contains useful variables
        self.assertEqual(list(widget.label_model),
                         [None, domain["with_extras"], domain["label"]])

        # `label` is chosen as default, and output has corresponding data
        self.assertIs(widget.label_variable, domain["label"])
        output = self.get_output(widget.Outputs.network)
        id_col = output.nodes.get_column("id")
        np.testing.assert_equal(id_col, np.arange(1, 8))

        # No variable, row matching. Error is shown and original labels are used
        widget._label_to_tabel = Mock(return_value=data[:7])
        widget.label_variable = None
        widget.label_changed()
        self.assertTrue(widget.Warning.mismatched_lengths.is_shown())
        output = self.get_output(widget.Outputs.network)
        self.assertIs(output.nodes, widget._label_to_tabel.return_value)

        # Choose a different variable; no error, output has corresponding data
        widget.label_variable = domain["with_extras"]
        widget.label_changed()
        self.assertFalse(widget.Warning.mismatched_lengths.is_shown())
        output = self.get_output(widget.Outputs.network)
        id_col = output.nodes.get_column("id")
        np.testing.assert_equal(id_col, np.arange(2, 9))

        # Remove data: model must be cleared, data back to original
        self.send_signal(widget.Inputs.items, None)
        self.assertEqual(list(widget.label_model), [None])
        output = self.get_output(widget.Outputs.network)
        self.assertIs(output.nodes, widget._label_to_tabel.return_value)

        # Bring data back, and turn on row matching;
        # this triggers an error; then remove data; error must be gone
        self.send_signal(widget.Inputs.items, data)
        widget.label_variable = None
        widget.label_changed()
        self.assertTrue(widget.Warning.mismatched_lengths.is_shown())

        self.send_signal(widget.Inputs.items, None)
        output = self.get_output(widget.Outputs.network)
        self.assertIs(output.nodes, widget._label_to_tabel.return_value)
        self.assertFalse(widget.Warning.mismatched_lengths.is_shown())

    def test_set_data_combo_setup(self):
        w = self.widget
        combo = w.controls.label_variable
        labs = [self.lab, self.lab2]

        for best in labs:
            for w.label_variable_hint in ["lab", "lab2", None]:
                with patch.object(w, "_vars_for_label",
                                  Mock(return_value=(best, labs))):
                    self.send_signal(w.Inputs.items, self.data)
                    self.assertEqual(list(combo.model()), [None] + labs)
                    self.assertEqual(w.label_variable.name,
                                     w.label_variable_hint or best.name)

        self.send_signal(w.Inputs.items, None)
        self.assertEqual(list(combo.model()), [None])
        self.assertIsNone(w.label_variable)

    def test_label_hints(self):
        w = self.widget
        combo = w.controls.label_variable

        self.send_signal(w.Inputs.items, self.data)
        ind_lab_2 = combo.model().indexOf(self.lab2)
        combo.setCurrentIndex(ind_lab_2)
        combo.activated[int].emit(ind_lab_2)

        self.send_signal(w.Inputs.items, self.edges)
        self.assertEqual(combo.currentIndex(), 0)
        self.assertIsNot(w.label_variable, self.lab2)
        assert ind_lab_2 != 0

        self.send_signal(w.Inputs.items, self.data)
        self.assertEqual(combo.currentIndex(), ind_lab_2)
        self.assertIs(w.label_variable, self.lab2)

    def test_vars_for_edges(self):
        self.widget.edges = Table.from_list(
            Domain([DiscreteVariable("discrete"),
                    ContinuousVariable("nonint"),
                    ContinuousVariable("has nans"),
                    ContinuousVariable("ok1")],
                   None,
                   [StringVariable("has missing"),
                    StringVariable("ok2"),
                    ContinuousVariable("ok3")
                    ]),
            [[0, 0, 0, 0, "a", "b", 0],
             [1, 0.5, np.nan, 1, "", "d", 2]]
        )
        *guess, edge_vars = self.widget._vars_for_edges()
        self.assertEqual([var.name for var in edge_vars], "ok1 ok2 ok3".split())
        self.assertEqual([var.name for var in guess], "ok1 ok3".split())

        self.widget.edges = Table.from_list(
            Domain([DiscreteVariable("discrete"),
                    ContinuousVariable("nonint"),
                    ContinuousVariable("has nans")
                    ],
                   None,
                   [StringVariable("has missing"),
                    StringVariable("ok1"),
                    ContinuousVariable("ok2"),
                    ContinuousVariable("ok3")
                    ]),
            [[0, 0, 0, "a", "b", 0, 0],
             [1, 0.5, np.nan, "", "d", 2, 1]]
        )
        *guess, edge_vars = self.widget._vars_for_edges()
        self.assertEqual([var.name for var in edge_vars], "ok1 ok2 ok3".split())
        self.assertEqual([var.name for var in guess], "ok2 ok3".split())

        self.widget.edges = Table.from_list(
            Domain([DiscreteVariable("discrete"),
                    ContinuousVariable("nonint"),
                    ContinuousVariable("has nans")
                    ],
                   None,
                   [StringVariable("has missing"),
                    StringVariable("ok1"),
                    ]),
            [[0, 0, 0, "a", "b"],
             [1, 0.5, np.nan, "", "d"]]
        )
        *guess, edge_vars = self.widget._vars_for_edges()
        self.assertEqual([var.name for var in edge_vars], ["ok1"])
        self.assertEqual(guess, [None, None])

        self.widget.edges = Table.from_list(
            Domain([DiscreteVariable("discrete"),
                    ContinuousVariable("nonint"),
                    ContinuousVariable("has nans")
                    ],
                   None,
                   [StringVariable("has missing"),
                    ]),
            [[0, 0, 0, "a"],
             [1, 0.5, np.nan, ""]]
        )
        *guess, edge_vars = self.widget._vars_for_edges()
        self.assertEqual([var.name for var in edge_vars], [])
        self.assertEqual(guess, [None, None])

    def test_set_edges_combo_setup(self):
        w = self.widget
        src = w.controls.edge_src_variable
        dst = w.controls.edge_dst_variable

        # Take the first two variables
        self.send_signal(w.Inputs.edges, self.edges)
        useful = "w src1 dst1 srcs dsts".split()
        self.assertEqual([var.name for var in src.model() if var], useful)
        self.assertEqual([var.name for var in dst.model() if var], useful)
        self.assertIs(w.edge_src_variable, self.w)
        self.assertIs(w.edge_dst_variable, self.src1)

        # Observe the hint
        w.edge_src_variable_hint, w.edge_dst_variable_hint = ("src1", "dst1")
        self.send_signal(w.Inputs.edges, self.edges)
        self.assertIs(w.edge_src_variable, self.src1)
        self.assertIs(w.edge_dst_variable, self.dst1)

        self.send_signal(w.Inputs.edges, None)
        self.assertIs(w.edge_src_variable, None)
        self.assertIs(w.edge_src_variable, None)

        # Remember the hint
        self.send_signal(w.Inputs.edges, self.edges)
        self.assertIs(w.edge_src_variable, self.src1)
        self.assertIs(w.edge_dst_variable, self.dst1)

        # Ignore the hint because one variable is missing; take the first
        # two of the same type
        dom = self.edges.domain
        self.send_signal(
            w.Inputs.edges,
            self.edges.transform(Domain(dom.attributes[2:], None, dom.metas)))
        self.assertIs(w.edge_src_variable, self.srcs)
        self.assertIs(w.edge_dst_variable, self.dsts)

        # Ignore the hint, and also set nothing else because there are no
        # two variables of the same type
        w.edge_src_variable_hint, w.edge_dst_variable_hint = ("src1", "dst1")
        dom = self.edges.domain
        self.send_signal(
            w.Inputs.edges,
            self.edges.transform(Domain(dom.attributes[2:], None, dom.metas[:1])))
        self.assertIsNone(w.edge_src_variable)
        self.assertIsNone(w.edge_dst_variable)

    def test_edge_hints(self):
        w = self.widget
        src = w.controls.edge_src_variable
        dst = w.controls.edge_dst_variable

        self.send_signal(w.Inputs.edges, self.edges)
        indesrc = src.model().indexOf(self.srcs)
        indedst = dst.model().indexOf(self.dsts)
        assert src.currentIndex() != indesrc and dst.currentIndex() != indedst
        select(src, self.srcs)
        select(dst, self.dsts)

        self.send_signal(w.Inputs.items, self.edges)
        self.assertEqual(src.currentIndex(), indesrc)
        self.assertEqual(dst.currentIndex(), indedst)

        self.send_signal(w.Inputs.edges, None)
        assert src.currentIndex() != indesrc and dst.currentIndex() != indedst

        self.send_signal(w.Inputs.edges, self.edges)
        self.assertEqual(src.currentIndex(), indesrc)
        self.assertEqual(dst.currentIndex(), indedst)

    @patch.object(OWNxFile, "compose_network")
    def test_call_compose(self, compose):
        w = self.widget

        compose.reset_mock()
        w.open_net_file(_get_test_net("test.net"))
        compose.assert_called_once()

        compose.reset_mock()
        self.send_signal(w.Inputs.items, self.data)
        compose.assert_called_once()

        compose.reset_mock()
        self.send_signal(w.Inputs.edges, self.edges)
        compose.assert_called_once()

        for combo in (w.controls.label_variable,
                      w.controls.edge_src_variable,
                      w.controls.edge_dst_variable):
            compose.reset_mock()
            combo.activated[int].emit(0)
            compose.assert_called_once()

    def test_network_nodes_no_data(self):
        w = self.widget
        w.original_network = self.network

        nodes = w.network_nodes()
        self.assertEqual(list(nodes.metas[:, 0]), self.network.nodes)
        self.assertTrue(w.Information.suggest_annotation.is_shown())
        self.assertFalse(w.Information.auto_annotation.is_shown())

        w.auto_data = self.data
        w.Information.suggest_annotation.clear()
        w.network_nodes()
        self.assertFalse(w.Information.suggest_annotation.is_shown())
        self.assertTrue(w.Information.auto_annotation.is_shown())

    def test_network_nodes_no_label_variable(self):
        w = self.widget
        w.original_network = self.network
        w.data = self.data
        nodes = w.network_nodes()
        np.testing.assert_equal(nodes.get_column("lab"),
                                self.data.get_column("lab"))
        np.testing.assert_equal(nodes.get_column("node_label"),
                                self.network.nodes)

    # also serves as `test_data_by_labels`
    def test_network_nodes_label_variable(self):
        w = self.widget
        # scramble node order
        w.original_network = \
            Network("bar qux foo bax baz".split(),
                    self.network.edges)
        self.send_signal(w.Inputs.items, self.data)
        w.label_variable = self.lab

        for nodes in (w.network_nodes(), w._data_by_labels(self.data)):
            np.testing.assert_equal(nodes.get_column("lab2"), list("bcaed"))
            self.assertEqual(nodes.domain, self.data.domain)

    def test_combined_data(self):
        w = self.widget
        w.original_network = self.network

        # Adds original graph's nodes as a column
        nodes = w._combined_data(self.data)
        np.testing.assert_equal(nodes.get_column("lab"), self.data.get_column("lab"))
        np.testing.assert_equal(nodes.get_column("node_label"), self.network.nodes)

        # Doesn't add a column of sequential numbers
        self.network.nodes = "1 2 3 4 5".split()
        nodes = w._combined_data(self.data)
        self.assertIs(nodes, self.data)

        # Doesn't add a column of sequential numbers
        self.network.nodes = "0 1 2 3 4".split()
        nodes = w._combined_data(self.data)
        self.assertIs(nodes, self.data)

        # Numbers, but not sequential: add a column
        self.network.nodes = "1 2 8 4 5".split()
        nodes = w._combined_data(self.data)
        np.testing.assert_equal(nodes.get_column("lab"), self.data.get_column("lab"))
        np.testing.assert_equal(nodes.get_column("node_label"), self.network.nodes)

        # Sequence of numbers, but not starting with 0 or 1: add a column
        self.network.nodes = "4 5 6 7 8".split()
        nodes = w._combined_data(self.data)
        np.testing.assert_equal(nodes.get_column("lab"), self.data.get_column("lab"))
        np.testing.assert_equal(nodes.get_column("node_label"), self.network.nodes)

    def test_label_to_tabel(self):
        w = self.widget
        w.original_network = self.network
        nodes = w._combined_data(self.data)
        np.testing.assert_equal(nodes.get_column("node_label"), self.network.nodes)

    def test_network_edges_no_data(self):
        w = self.widget
        w.original_network = self.network

        self.assertIs(w.network_edges()[0], self.network.edges[0])

        self.send_signal(w.Inputs.edges, self.edges)
        assert w.network_edges()[0] is not self.network.edges[0]

        w.edge_src_variable = None
        self.assertIs(w.network_edges(), self.network.edges)

        w.edge_src_variable = w.edge_dst_variable
        w.edge_dst_variable = None
        self.assertIs(w.network_edges(), self.network.edges)

    def test_network_edges_resorting(self):
        w = self.widget
        w.original_network = self.network
        w.edge_src_variable_hint = "srcs"
        w.edge_dst_variable_hint = "dsts"

        # properly sort the edge table data
        self.send_signal(w.Inputs.edges, self.edges)
        edges = w.network_edges()[0]
        np.testing.assert_equal(edges.edge_data.get_column("w"),
                                [2, 1, 8, 3])
        self.assertFalse(w.Warning.missing_edges.is_shown())
        self.assertFalse(w.Warning.extra_edges.is_shown())

        # missing data
        self.send_signal(w.Inputs.edges, self.edges[1:])
        edges = w.network_edges()[0]
        np.testing.assert_equal(edges.edge_data.get_column("w"),
                                [2, np.nan, 8, 3])
        self.assertTrue(w.Warning.missing_edges.is_shown())
        self.assertFalse(w.Warning.extra_edges.is_shown())
        w.Warning.missing_edges.clear()

        # missing and extra
        with self.edges.unlocked(self.edges.metas):
            self.edges.metas[0, 1] = "bax"
        self.send_signal(w.Inputs.edges, self.edges)
        self.assertTrue(w.Warning.missing_edges.is_shown())
        self.assertTrue(w.Warning.extra_edges.is_shown())
        w.Warning.missing_edges.clear()
        w.Warning.extra_edges.clear()

        # non-existing label
        with self.edges.unlocked(self.edges.metas):
            self.edges.metas[0, 1] = "no such label"
        self.send_signal(w.Inputs.edges, self.edges)
        self.assertTrue(w.Warning.missing_edges.is_shown())
        self.assertTrue(w.Warning.extra_edges.is_shown())

    def test_network_edge_directed(self):
        undirected = self.network
        directed = Network(self.network.nodes,
                           [DirectedEdges(self.network.edges[0].edges)])
        assert not undirected.edges[0].directed

        def all_matched(net):
            w.original_network = net
            return not np.any(np.isnan(w.network_edges()[0].edge_data.get_column(0)))

        w = self.widget
        w.original_network = self.network

        self.send_signal(w.Inputs.edges, self.edges)
        w.edge_src_variable = self.srcs
        w.edge_dst_variable = self.dsts
        self.assertTrue(all_matched(undirected))
        self.assertTrue(all_matched(directed))

        w.edge_src_variable = self.dsts
        w.edge_dst_variable = self.srcs
        self.assertTrue(all_matched(undirected))
        self.assertFalse(all_matched(directed))

    def test_network_from_inputs_no_edge_data(self):
        w = self.widget
        w.original_network = self.network
        w.edge_src_variable_hint = "srcs"
        w.edge_dst_variable_hint = "dsts"

        self.send_signal(w.Inputs.items, self.data)

        self.assertIsNone(w.network_from_inputs())

        self.send_signal(w.Inputs.edges, self.edges)
        assert w.network_from_inputs() is not None

        w.edge_src_variable = None
        self.assertIsNone(w.network_from_inputs())

        w.edge_src_variable = w.edge_dst_variable
        w.edge_dst_variable = None
        self.assertIsNone(w.network_from_inputs())

    def test_network_from_inputs(self):
        w = self.widget
        w.original_network = self.network
        w.edge_src_variable_hint = "srcs"
        w.edge_dst_variable_hint = "dsts"
        self.send_signal(w.Inputs.edges, self.edges)

        with patch("orangecontrib.network.network.compose.network_from_edge_table") as m:
            self.assertIs(w.network_from_inputs(), m.return_value)
            m.assert_called_once_with(self.edges, self.srcs, self.dsts)

        self.send_signal(w.Inputs.items, self.data)
        with patch("orangecontrib.network.network.compose.network_from_tables") as m:
            self.assertIs(w.network_from_inputs(), m.return_value)
            m.assert_called_once_with(self.data, w.label_variable,
                                      self.edges, self.srcs, self.dsts)

    def test_network_from_inputs_errors(self):
        w = self.widget
        w.original_network = self.network
        w.edge_src_variable_hint = "srcs"
        w.edge_dst_variable_hint = "dsts"
        self.send_signal(w.Inputs.edges, self.edges)
        self.send_signal(w.Inputs.items, self.data)
        w.label_variable = None

        def assert_shown(exp):
            for err in (w.Error.no_label_variable,
                        w.Error.missing_label_values,
                        w.Error.mismatched_edge_variables,
                        w.Error.unidentified_nodes):
                self.assertIs(err.is_shown(), err is exp, repr(err))
                err.clear()

        self.assertIsNone(w.network_from_inputs())
        assert_shown(w.Error.no_label_variable)

        d = Table.concatenate(
            [self.data,
             Table.from_list(self.data.domain, [[1, "", "a"]])
             ])
        self.send_signal(w.Inputs.items, d)
        w.label_variable = self.lab
        self.assertIsNone(w.network_from_inputs())
        assert_shown(w.Error.missing_label_values)

        self.send_signal(w.Inputs.items, self.data)
        w.edge_src_variable = self.src1
        w.edge_dst_variable = self.dsts
        self.assertIsNone(w.network_from_inputs())
        assert_shown(w.Error.mismatched_edge_variables)

        e = Table.concatenate(
            [self.edges,
             Table.from_list(self.edges.domain,
                             [[1, 2, 3, 4, "boo", "far"]])]
        )
        self.send_signal(w.Inputs.edges, e)
        self.assertIsNone(w.network_from_inputs())
        assert_shown(w.Error.unidentified_nodes)

    def test_source_radios(self):
        w = self.widget
        radio = w.controls.original_net_source.group
        test_labels = "aaa bbb ccc ddd eee fff ggg".split()
        etest_labels = "bar bax baz foo".split()
        w.original_net_source = w.LoadFromFile
        w.edge_src_variable_hint = "srcs"
        w.edge_dst_variable_hint = "dsts"

        def rchoose(opt):
            radio.button(opt).click()

        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=(_get_test_net("test.net"), ".net")):
            w.select_net_file()
        out = self.get_output(w.Outputs.network)
        np.testing.assert_equal(out.nodes.get_column(-1), test_labels)

        self.send_signal(w.Inputs.edges, self.edges)
        out = self.get_output(w.Outputs.network)
        np.testing.assert_equal(out.nodes.get_column(-1), test_labels)

        rchoose(w.ConstructFromInputs)
        out = self.get_output(w.Outputs.network)
        np.testing.assert_equal(out.nodes.get_column(-1), etest_labels)

        rchoose(w.LoadFromFile)
        out = self.get_output(w.Outputs.network)
        np.testing.assert_equal(out.nodes.get_column(-1), test_labels)

        rchoose(w.ConstructFromInputs)
        out = self.get_output(w.Outputs.network)
        np.testing.assert_equal(out.nodes.get_column(-1), etest_labels)

        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=(self._get_filename("davis.net"), ".net")):
            w.browse_net_file()
        self.assertEqual(w.original_net_source, w.LoadFromFile)
        out = self.get_output(w.Outputs.network)
        np.testing.assert_equal(out.nodes.get_column(-1)[:3],
                                ['EVELYN', 'LAURA', 'THERESA'])

    def test_flow(self):
        w = self.widget
        src = w.controls.edge_src_variable
        dst = w.controls.edge_dst_variable

        graph_order = [2, 1, 8, 3]
        rev_index_order = [2, 3, 1, 8]
        alpha_order = [3, 8, 2, 1]

        w.send_report()

        w.open_net_file(_get_test_net("test-compose.net"))

        # Just input file: no label or edge data, labels come from the file
        out = self.get_output(w.Outputs.network)
        self.assertEqual(len(out.nodes.domain.attributes), 0)
        self.assertEqual(len(out.nodes.domain.class_vars), 0)
        self.assertEqual(len(out.nodes.domain.metas), 1)
        np.testing.assert_equal(out.nodes.get_column(-1), self.data.get_column("lab"))
        self.assertIsNone(out.edges[0].edge_data)
        w.send_report()

        # Input file + data: we have node data, but still no edge data
        self.send_signal(w.Inputs.items, self.data)
        out = self.get_output(w.Outputs.network)
        self.assertEqual(out.nodes.domain.attributes, self.data.domain.attributes)
        self.assertEqual(out.nodes.domain.class_vars, self.data.domain.class_vars)
        self.assertEqual(out.nodes.domain.metas, self.data.domain.metas)
        np.testing.assert_equal(out.nodes.get_column(0), self.data.get_column(0))
        self.assertIsNone(out.edges[0].edge_data)
        w.send_report()

        # Input file + data + edge data
        self.send_signal(w.Inputs.edges, self.edges)
        out = self.get_output(w.Outputs.network)
        self.assertEqual(out.nodes.domain.attributes, self.data.domain.attributes)
        self.assertEqual(out.nodes.domain.class_vars, self.data.domain.class_vars)
        self.assertEqual(out.nodes.domain.metas, self.data.domain.metas)
        np.testing.assert_equal(out.nodes.get_column(0), self.data.get_column(0))
        # edge data exists but is nan because wrong columns are chosen
        self.assertIsNotNone(out.edges[0].edge_data)
        w.send_report()

        # now edge data needs to be properly permuted
        select(src, self.srcs)
        select(dst, self.dsts)
        out = self.get_output(w.Outputs.network)
        np.testing.assert_equal(out.edges[0].edge_data.get_column(0), [2, 1, 8, 3])
        w.send_report()

        # Input file + edge data
        self.send_signal(w.Inputs.items, None)
        out = self.get_output(w.Outputs.network)
        self.assertEqual(len(out.nodes.domain.attributes), 0)
        self.assertEqual(len(out.nodes.domain.class_vars), 0)
        self.assertEqual(len(out.nodes.domain.metas), 1)
        np.testing.assert_equal(out.nodes.get_column(-1), self.data.get_column("lab"))
        np.testing.assert_equal(out.edges[0].edge_data.get_column(0), [2, 1, 8, 3])
        w.send_report()

        # Just edge data; no labels or network from file
        w.open_net_file(None)
        out = self.get_output(w.Outputs.network)
        self.assertEqual(len(out.nodes.domain.attributes), 0)
        self.assertEqual(len(out.nodes.domain.class_vars), 0)
        self.assertEqual(len(out.nodes.domain.metas), 1)
        np.testing.assert_equal(out.nodes.get_column(-1), "bar bax baz foo".split())
        w.send_report()

        edom = out.edges[0].edge_data.domain
        self.assertEqual(edom.attributes, self.edges.domain.attributes)
        self.assertEqual(len(edom.class_vars), 0)
        self.assertEqual(len(edom.metas), 0)
        # ordered alphabetically by labels
        np.testing.assert_equal(out.edges[0].edge_data.get_column(0), alpha_order)
        w.send_report()

        # Just edge data, but using indices; no labels or network from file
        select(src, self.dst1)  # switch to have a different order
        select(dst, self.src1)
        out = self.get_output(w.Outputs.network)
        self.assertEqual(len(out.nodes.domain.attributes), 0)
        self.assertEqual(len(out.nodes.domain.class_vars), 0)
        self.assertEqual(len(out.nodes.domain.metas), 1)
        np.testing.assert_equal(out.nodes.get_column(-1), "1 2 3 4 5".split())
        edom = out.edges[0].edge_data.domain
        self.assertEqual(len(edom.attributes), 2)  # no used attrs
        self.assertEqual(len(edom.class_vars), 0)
        self.assertEqual(len(edom.metas), 2)
        # ordered indices; dst1 first
        np.testing.assert_equal(out.edges[0].edge_data.get_column(0), rev_index_order)
        w.send_report()

        # Label and edge data, no network from file
        self.send_signal(w.Inputs.items, self.data)
        out = self.get_output(w.Outputs.network)
        self.assertIs(out.nodes, self.data)
        self.assertEqual(len(edom.attributes), 2)  # no used attrs
        self.assertEqual(len(edom.class_vars), 0)
        self.assertEqual(len(edom.metas), 2)
        # ordered indices; dst1 first
        np.testing.assert_equal(out.edges[0].edge_data.get_column(0), rev_index_order)
        w.send_report()

        select(src, self.srcs)
        select(dst, self.dsts)
        out = self.get_output(w.Outputs.network)
        np.testing.assert_equal(out.edges[0].edge_data.get_column(0), graph_order)
        w.send_report()

        # Label data, no edges or network from file
        self.send_signal(w.Inputs.edges, None)
        out = self.get_output(w.Outputs.network)
        self.assertIsNone(out)
        w.send_report()

        # Kaput
        self.send_signal(w.Inputs.items, None)
        out = self.get_output(w.Outputs.network)
        self.assertIsNone(out)
        w.send_report()

    def test_migrate_from_context_settings(self):
        settings = {'__version__': 1,
                    'context_settings': [
                        {'values': {'__version__': 1},
                         'useful_vars': {'artist', 'albums'},
                         'label_variable': 'artist'}]}
        w = self.create_widget(OWNxFile, stored_settings=settings)
        self.assertEqual(w.label_variable_hint, "artist")

        settings = {'__version__': 1,
                    'context_settings': []}
        w = self.create_widget(OWNxFile, stored_settings=settings)
        self.assertIsNone(w.label_variable_hint)

        settings = {'__version__': 1,
                    'label_variable_hint': "foo"}
        w = self.create_widget(OWNxFile, stored_settings=settings)
        self.assertEqual(w.label_variable_hint, "foo")


if __name__ == "__main__":
    unittest.main()
