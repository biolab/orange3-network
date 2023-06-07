from operator import itemgetter
from os import path
from typing import Optional

from itertools import product, chain, count
from traceback import format_exception_only

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QStyle, QSizePolicy, QFileDialog

from Orange.util import get_entry_point
from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.data.util import get_unique_names
from Orange.widgets import gui
from Orange.widgets.report import bool_str
from orangewidget.settings import Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from orangecontrib.network.network import Network, compose
from orangecontrib.network.network.readwrite import read_pajek


demos_path = next(
    get_entry_point("Orange3-Network", "orange.data.io.search_paths", "network")
    ())[1]


# TODO: Check box whether the network constructed from edges input is directed


class OWNxFile(OWWidget):
    name = "Network File"
    description = "Read network graph file"
    icon = "icons/NetworkFile.svg"
    priority = 6410

    class Inputs:
        items = Input("Items", Table)
        edges = Input("Edges", Table)

    class Outputs:
        network = Output("Network", Network)
        items = Output("Items", Table)

    LoadFromFile, ConstructFromInputs = 0, 1

    label_variable_hint: Optional[str] = Setting(None, schema_only=True)
    edge_src_variable_hint: Optional[str] = Setting(None, schema_only=True)
    edge_dst_variable_hint: Optional[str] = Setting(None, schema_only=True)
    original_net_source = Setting(LoadFromFile, schema_only=True)
    recentFiles = Setting([])

    class Information(OWWidget.Information):
        auto_annotation = Msg(
            'Nodes annotated with data from file with the same name')
        suggest_annotation = Msg(
            'Add optional data input to annotate nodes')

    class Warning(OWWidget.Warning):
        missing_edges = Msg('There is no data for some edges')
        extra_edges = Msg(
            'Edge data contains data for some edges that do not exist')
        mismatched_lengths = Msg(
            "Data for nodes is ignored because its size does not match "
            "the number of nodes:\n"
            "select a data column whose values can be matched with network "
            "labels")

    class Error(OWWidget.Error):
        io_error = Msg('Error reading file "{}"\n{}')
        error_parsing_file = Msg('Error reading file "{}"')
        auto_data_failed = Msg(
            "Attempt to read {} failed\n"
            "The widget tried to annotated nodes with data from\n"
            "a file with the same name.")
        no_label_variable = Msg(
            "Choose a label column to construct a network from tables")
        mismatched_edge_variables = Msg(
            "Source and destination columns must be of the same type\n"
            "(numerical or text)"
        )
        unidentified_nodes = Msg(
            "Edge data refers to nodes that do not exist in the node data:\n{}"
        )
        missing_label_values = Msg(
            "Constructing a network from tables requires a label column "
            "without missing values"
        )

    want_main_area = False
    mainArea_width_height_ratio = None

    def __init__(self):
        super().__init__()

        self.network = None
        self.auto_data = None
        self.original_network = None
        self.data = None
        self.label_variable = None
        self.edges = None
        self.edge_src_variable = self.edge_dst_variable = None
        self.net_index = 0

        vb = gui.radioButtons(
            self.controlArea, self, "original_net_source",
            box="Network from file",
            callback=self.on_source_changed)
        hb = gui.hBox(vb)
        gui.appendRadioButton(
            vb, "Load network from file: ", insertInto=hb, id=self.LoadFromFile,
            tooltip="Load from file and "
                    "use inputs (if any) to annotate vertices and edges")
        self.filecombo = gui.comboBox(
            hb, self, "net_index", callback=self.select_net_file,
            minimumWidth=250)
        gui.button(
            hb, self, '...', callback=self.browse_net_file, disabled=0,
            icon=self.style().standardIcon(QStyle.SP_DirOpenIcon),
            sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Fixed))
        gui.button(
            hb, self, 'Reload', callback=self.reload,
            icon=self.style().standardIcon(QStyle.SP_BrowserReload),
            sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Fixed))
        hb = gui.hBox(vb)
        gui.appendRadioButton(
            vb, "Construct network from input tables", insertInto=hb,
            id=self.ConstructFromInputs
        )

        self.label_model = VariableListModel(placeholder="(Match by rows)")
        self.label_model[:] = [None]
        gui.comboBox(
            self.controlArea, self, "label_variable",
            box="Node description (from input signal)",
            label="Match node labels with values of ", orientation=Qt.Horizontal,
            model=self.label_model, callback=self.label_changed)

        self.edge_model = VariableListModel(placeholder="(None)")
        box = gui.vBox(self.controlArea,
                       box="Edge description (from input signal)")
        ebox = gui.hBox(box)
        gui.comboBox(
            ebox, self, "edge_src_variable", label="Source node label:",
            model=self.edge_model, callback=self.edge_changed)
        gui.separator(ebox, 16)
        gui.comboBox(
            ebox, self, "edge_dst_variable", label="Destination node label:",
            model=self.edge_model, callback=self.edge_changed)

        self.update_file_combo()
        self.setFixedHeight(self.sizeHint().height())
        self.reload()

    def update_file_combo(self):
        self.filecombo.clear()
        for file in self.recentFiles or ("(None)",):
            self.filecombo.addItem(path.basename(file))
        self.filecombo.addItem("Browse documentation networks...")
        self.filecombo.updateGeometry()

    def browse_net_file(self, browse_demos=False):
        """user pressed the '...' button to manually select a file to load"""
        if browse_demos:
            startfile = demos_path
        else:
            startfile = self.recentFiles[0] if self.recentFiles else '.'

        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open a Network File', startfile,
            ';;'.join(("Pajek files (*.net *.pajek)",)))
        if not filename:
            return False

        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        self.recentFiles.insert(0, filename)

        self.update_file_combo()
        self.net_index = 0
        self.select_net_file()
        return True

    def reload(self):
        if self.recentFiles:
            self.select_net_file()

    def on_source_changed(self):
        if self.original_net_source == self.LoadFromFile:
            if self.recentFiles and self.net_index < len(self.recentFiles):
                self.open_net_file(self.recentFiles[0])
            else:
                self.select_net_file()
        else:
            self.open_net_file(None)

    def select_net_file(self):
        """user selected a graph file from the combo box"""
        self.original_net_source = self.LoadFromFile
        if self.net_index > len(self.recentFiles) - 1:
            if not self.browse_net_file(True):
                if self.net_index >= len(self.recentFiles):
                    self.original_net_source = self.ConstructFromInputs
                return  # Cancelled
        elif self.net_index:
            self.recentFiles.insert(0, self.recentFiles.pop(self.net_index))
            self.net_index = 0
            self.update_file_combo()
        if self.recentFiles:
            self.open_net_file(self.recentFiles[0])

    def open_net_file(self, filename):
        """Read network from file."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self.original_network = None
        self.auto_data = None
        if filename is not None:
            try:
                self.original_network = read_pajek(filename)
            except OSError as err:
                self.Error.io_error(
                    filename,
                    "".join(format_exception_only(type(err), err)).rstrip())
            except Exception:  # pylint: disable=broad-except
                self.Error.error_parsing_file(filename)
            else:
                self.read_auto_data(filename)
        else:
            self.original_net_source = self.ConstructFromInputs
        self.compose_network()

    def read_auto_data(self, filename):
        self.Error.auto_data_failed.clear()

        self.auto_data = None
        errored_file = None
        basenames = (filename,
                     path.splitext(filename)[0],
                     path.splitext(filename)[0] + '_items')
        for basename, ext in product(basenames, ('.tab', '.tsv', '.csv')):
            filename = basename + ext
            if path.exists(filename):
                try:
                    self.auto_data = Table.from_file(filename)
                    break
                except Exception:  # pylint: disable=broad-except
                    errored_file = filename
        else:
            if errored_file:
                self.Error.auto_data_failed(errored_file)

    @property
    def original_nodes(self):
        return self.original_network.nodes

    @Inputs.items
    def set_data(self, data):
        self.data = data
        data = self.data if self.data is not None else self.auto_data
        if data is None:
            self.label_model[:] = [None]
            self.label_variable = None
            return

        best_var, useful_vars = self._vars_for_label()
        self.label_model[:] = [None] + useful_vars
        self.label_variable = \
            self._find_variable(self.label_variable_hint, useful_vars, best_var)

    @staticmethod
    def _find_variable(name, variables, default=None):
        for var in variables:
            if var.name == name:
                return var
        return default

    def _vars_for_label(self):
        useful_vars = []
        overs = []
        if self.original_network is not None:
            original_nodes = set(self.original_nodes)
        else:
            original_nodes = set()

        for var in self.data.domain.metas:
            if not isinstance(var, StringVariable):
                continue
            values = self.data.get_column(var)
            values = values[values != ""]
            set_values = set(values)
            # if you remove the subset condition, also change data_by_labels
            if len(values) != len(set_values) \
                    or not original_nodes <= set_values:
                continue
            useful_vars.append(var)
            overs.append(len(set_values - original_nodes))
        if not useful_vars:
            return None, []
        best_var = useful_vars[np.argmin(overs)]
        return best_var, useful_vars

    @Inputs.edges
    def set_edges(self, edges):
        self.edges = edges
        if self.edges is None:
            self.edge_model[:] = [None]
            self.edge_src_variable = self.edge_dst_variable = None
            return

        *guess, edge_vars = self._vars_for_edges()
        src = self._find_variable(self.edge_src_variable_hint, edge_vars)
        dst = self._find_variable(self.edge_dst_variable_hint, edge_vars)
        if not (src and dst):
            src, dst = guess
        self.edge_model[:] = [None] + edge_vars
        self.edge_src_variable, self.edge_dst_variable = src, dst

    def _vars_for_edges(self):
        edges = self.edges
        useful_vars = [
            var for var in chain(edges.domain.variables, edges.domain.metas)
            if (var.is_string
                and np.all(edges.get_column(var) != ""))
            or (type(var) is ContinuousVariable
                and not np.isnan(np.sum(col := edges.get_column(var)))
                and np.all(np.modf(col)[0] == 0))
            ]
        if len(useful_vars) < 2:
            src = dst = None
        else:
            # Take the first variable and find the next of the same type
            src = useful_vars[0]
            for dst in useful_vars[1:]:
                if type(src) is type(dst):
                    break
            else:
                # If there's none, the second and third are of the same type
                if len(useful_vars) >= 3:
                    src, dst = tuple(useful_vars[1:3])
                else:
                    src = dst = None
        return src, dst, useful_vars

    def handleNewSignals(self):
        self.compose_network()

    def label_changed(self):
        self.label_variable_hint = self._hint_for(self.label_variable)
        self.compose_network()

    def edge_changed(self):
        self.edge_src_variable_hint = self._hint_for(self.edge_src_variable)
        self.edge_dst_variable_hint = self._hint_for(self.edge_dst_variable)
        self.compose_network()

    @staticmethod
    def _hint_for(var):
        return var and var.name

    def compose_network(self):
        self.Error.no_label_variable.clear()
        self.Error.mismatched_edge_variables.clear()
        self.Error.unidentified_nodes.clear()
        self.Error.missing_label_values.clear()
        self.Warning.mismatched_lengths.clear()
        self.Warning.missing_edges.clear()
        self.Warning.extra_edges.clear()
        self.Information.suggest_annotation.clear()
        self.Information.auto_annotation.clear()

        if self.original_network is not None:
            self.network = self.annotated_read_network()
        elif self.original_net_source == self.ConstructFromInputs and self.edges:
            self.network = self.network_from_inputs()
        else:
            self.network = None
        self.send_output()

    def annotated_read_network(self):
        assert self.original_network is not None

        return Network(
            self.network_nodes(), self.network_edges(),
            self.original_network.name,
            self.original_network.coordinates)

    def network_nodes(self):
        assert self.original_network is not None

        data = self.data
        if data is None:
            if self.auto_data is not None:
                data = self.auto_data
                self.Information.auto_annotation()
            else:
                self.Information.suggest_annotation()
        if data is not None \
                and self.label_variable is None \
                and len(data) != self.original_network.number_of_nodes():
            self.Warning.mismatched_lengths()
            data = None

        if data is None:
            return self._label_to_tabel()
        elif self.label_variable is None:
            return self._combined_data(data)
        else:
            return self._data_by_labels(data)

    def _data_by_labels(self, data):
        """
        Return data rearranged so that values of `self.label_variable`
        match the original graph labels.
        """
        # all node labels exist in data_col; this is ensured by _vars_for_label
        data_col = data.get_column(self.label_variable)
        data_rows = {label: row for row, label in enumerate(data_col)}
        indices = [data_rows[label] for label in self.original_nodes]
        return data[indices]

    def _combined_data(self, source: Table):
        """
        Return `source` with an additional column `node_label` containing
        original graph labels.
        If original labels are sequential numbers starting with 0 or 1,
        just return `source`.
        """
        nodes = np.array(self.original_nodes, dtype=str)
        try:
            nums = [int(x) for x in nodes]
        except ValueError:
            pass
        else:
            nums = np.sort(nums)
            if nums[0] in (0, 1) and np.all(nums[1:] - nums[:-1] == 1):
                return source
        return source.add_column(
            StringVariable(get_unique_names(source.domain, "node_label")),
            nodes)

    def _label_to_tabel(self):
        """
        Return a data table containing the graphs original node labels
        """
        domain = Domain([], [], [StringVariable("node_label")])
        n = len(self.original_nodes)
        data = Table.from_numpy(
            domain, np.empty((n, 0)), np.empty((n, 0)),
            np.array(self.original_nodes, dtype=str).reshape(-1, 1))
        return data

    def network_edges(self):
        if self.edges is None or (
                self.edge_src_variable is None
                or self.edge_dst_variable is None):
            return self.original_network.edges

        src_col = self.edges.get_column(self.edge_src_variable)
        dst_col = self.edges.get_column(self.edge_dst_variable)
        edge_data = {
            (src, dst): row for row, src, dst in zip(count(), src_col, dst_col)}
        edges0 = self.original_network.edges[0]
        directed = edges0.directed
        edges = edges0.edges
        n_edges = len(self.edges)
        edges.sort_indices()
        indices = []
        for src_idx, ptr_begin, ptr_end in zip(count(), edges.indptr, edges.indptr[1:]):
            for dst_idx in edges.indices[ptr_begin:ptr_end]:
                src_node = self.original_nodes[src_idx]
                dst_node = self.original_nodes[dst_idx]
                ind = edge_data.pop((src_node, dst_node), n_edges)
                if ind == n_edges and not directed:
                    ind = edge_data.pop((dst_node, src_node),n_edges)
                indices.append(ind)
        if edge_data:
            self.Warning.extra_edges()

        domain = self.edges.domain
        edge_attrs = (self.edge_src_variable, self.edge_dst_variable)
        pure_domain = Domain(
            *([var for var in part if var not in edge_attrs]
            for part in (domain.attributes, domain.class_vars, domain.metas)))
        pure_table = self.edges.transform(pure_domain)
        if np.max(indices) == n_edges:
            extra_row = Table.from_list(
                pure_domain, [[np.nan] * (len(pure_domain.variables) + len(pure_domain.metas))]
            )
            pure_table = Table.concatenate((pure_table, extra_row))
            self.Warning.missing_edges()

        edges = self.original_network.edges
        return [
            type(edges[0])(edges[0].edges, pure_table[indices], edges[0].name)
        ] + edges[1:]

    def network_from_inputs(self):
        if self.edges is None \
                or self.edge_src_variable is None \
                or self.edge_dst_variable is None:
            return None

        try:
            if self.data is None:
                return compose.network_from_edge_table(
                    self.edges, self.edge_src_variable, self.edge_dst_variable)
            else:
                if self.label_variable is None:
                    self.Error.no_label_variable()
                    return None
                if np.any(self.data.get_column(self.label_variable) == ""):
                    self.Error.missing_label_values()
                    return
                return compose.network_from_tables(
                    self.data, self.label_variable,
                    self.edges, self.edge_src_variable, self.edge_dst_variable)
        except compose.MismatchingEdgeVariables:
            self.Error.mismatched_edge_variables()
        except compose.UnknownNodes as exc:
            msg = str(exc)
            self.Error.unidentified_nodes(msg[:60] + "..." * (len(msg) > 60))
        # We intentionally don't handle `compose.NonUniqueLabels`:
        # the widget should prevent it by not allowing to select such variables
        return None

    def send_output(self):
        self.Outputs.network.send(self.network)
        self.Outputs.items.send(self.network and self.network.nodes)

    def send_report(self):
        file_data = [("File name", self.filecombo.currentText())]
        if self.original_network:
            file_data += [
                ("Vertices", self.network.number_of_nodes()),
                ("Edges", self.network.number_of_edges()),
                ("Directed", bool_str(self.network.edges[0].directed))
            ]
        self.report_items("Network file", file_data)

        ctrl = self.controls
        annotation = []
        if self.data is not None:
            annotation += [
                ("Table with vertex data", self.data.name),
                ("Column with label", ctrl.label_variable.currentText())
            ]
        if self.edges is not None:
            annotation += [
                ("Table with edge data", self.edges.name),
                ("Columns for matching with label",
                f"{ctrl.edge_src_variable.currentText()} and "
                f"{ctrl.edge_dst_variable.currentText()}")
            ]
        if annotation:
            self.report_items("Additional data from inputs", annotation)

    @classmethod
    def migrate_settings(cls, settings, version):
        if "context_settings" in settings:
            settings["label_variable_hint"] = None
            if len(settings["context_settings"]) > 0:
                context = settings["context_settings"][-1]
                if "label_variable" in context:
                    settings["label_variable_hint"] = context["label_variable"]
            del settings["context_settings"]


if __name__ == "__main__":
    WidgetPreview(OWNxFile).run(set_edges=Table("heart_disease"))
