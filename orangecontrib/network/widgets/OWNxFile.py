from os import path
from itertools import product
from traceback import format_exception_only

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QStyle, QSizePolicy, QFileDialog

from Orange.util import get_entry_point
from Orange.data import Table, Domain, StringVariable
from Orange.data.util import get_unique_names
from Orange.widgets import gui, settings
from Orange.widgets.settings import ContextHandler
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from orangecontrib.network.network import Network
from orangecontrib.network.network.readwrite import read_pajek


class NxFileContextHandler(ContextHandler):
    def new_context(self, useful_vars):
        context = super().new_context()
        context.useful_vars = {var.name for var in useful_vars}
        context.label_variable = None
        return context

    # noinspection PyMethodOverriding
    def match(self, context, useful_vars):
        useful_vars = {var.name for var in useful_vars}
        if context.useful_vars == useful_vars:
            return self.PERFECT_MATCH
        # context.label_variable can also be None; this would always match,
        # so ignore it
        elif context.label_variable in useful_vars:
            return self.MATCH
        else:
            return self.NO_MATCH

    def settings_from_widget(self, widget, *_):
        context = widget.current_context
        if context is not None:
            context.label_variable = \
                widget.label_variable and widget.label_variable.name

    def settings_to_widget(self, widget, useful_vars):
        context = widget.current_context
        widget.label_variable = None
        if context.label_variable is not None:
            for var in useful_vars:
                if var.name == context.label_variable:
                    widget.label_variable = var
                    break


demos_path = next(
    get_entry_point("Orange3-Network", "orange.data.io.search_paths", "network")
    ())[1]


class OWNxFile(OWWidget):
    name = "Network File"
    description = "Read network graph file"
    icon = "icons/NetworkFile.svg"
    priority = 6410

    class Inputs:
        items = Input("Items", Table)

    class Outputs:
        network = Output("Network", Network)
        items = Output("Items", Table)

    settingsHandler = NxFileContextHandler()
    label_variable: StringVariable = settings.ContextSetting(None)
    recentFiles = settings.Setting([])

    class Information(OWWidget.Information):
        auto_annotation = Msg(
            'Nodes annotated with data from file with the same name')
        suggest_annotation = Msg(
            'Add optional data input to annotate nodes')

    class Error(OWWidget.Error):
        io_error = Msg('Error reading file "{}"\n{}')
        error_parsing_file = Msg('Error reading file "{}"')
        auto_data_failed = Msg(
            "Attempt to read {} failed\n"
            "The widget tried to annotated nodes with data from\n"
            "a file with the same name.")
        mismatched_lengths = Msg(
            "Data size does not match the number of nodes.\n"
            "Select a data column whose values can be matched with network "
            "labels")

    want_main_area = False
    mainArea_width_height_ratio = None

    def __init__(self):
        super().__init__()

        self.network = None
        self.auto_data = None
        self.original_nodes = None
        self.data = None
        self.net_index = 0

        hb = gui.widgetBox(self.controlArea, orientation=Qt.Horizontal)
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

        self.label_model = VariableListModel(placeholder="(Match by rows)")
        self.label_model[:] = [None]
        gui.comboBox(
            self.controlArea, self, "label_variable", box=True,
            label="Match node labels to data column: ", orientation=Qt.Horizontal,
            model=self.label_model, callback=self.label_changed)

        self.populate_comboboxes()
        self.setFixedHeight(self.sizeHint().height())
        self.reload()

    @Inputs.items
    def set_data(self, data):
        self.data = data
        self.update_label_combo()
        self.send_output()

    def populate_comboboxes(self):
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

        self.populate_comboboxes()
        self.net_index = 0
        self.select_net_file()
        return True

    def reload(self):
        if self.recentFiles:
            self.select_net_file()

    def select_net_file(self):
        """user selected a graph file from the combo box"""
        if self.net_index > len(self.recentFiles) - 1:
            if not self.browse_net_file(True):
                return  # Cancelled
        elif self.net_index:
            self.recentFiles.insert(0, self.recentFiles.pop(self.net_index))
            self.net_index = 0
            self.populate_comboboxes()
        if self.recentFiles:
            self.open_net_file(self.recentFiles[0])

    def open_net_file(self, filename):
        """Read network from file."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self.network = None
        self.original_nodes = None
        try:
            self.network = read_pajek(filename)
        except OSError as err:
            self.Error.io_error(
                filename,
                "".join(format_exception_only(type(err), err)).rstrip())
        except Exception:  # pylint: disable=broad-except
            self.Error.error_parsing_file(filename)
        else:
            self.original_nodes = self.network.nodes
            self.read_auto_data(filename)
        self.update_label_combo()
        self.send_output()

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

    def update_label_combo(self):
        self.closeContext()
        data = self.data if self.data is not None else self.auto_data
        if self.network is None or data is None:
            self.label_model[:] = [None]
        else:
            best_var, useful_vars = self._vars_for_label(data)
            self.label_model[:] = [None] + useful_vars
            self.label_variable = best_var
            self.openContext(useful_vars)
        self.set_network_nodes()

    def _vars_for_label(self, data: Table):
        vars_and_overs = []
        original_nodes = set(self.original_nodes)
        for var in data.domain.metas:
            if not isinstance(var, StringVariable):
                continue
            values, _ = data.get_column_view(var)
            values = values[values != ""]
            set_values = set(values)
            if len(values) != len(set_values) \
                    or not original_nodes <= set_values:
                continue
            vars_and_overs.append((len(set_values - original_nodes), var))
        if not vars_and_overs:
            return None, []
        _, best_var = min(vars_and_overs)
        useful_string_vars = [var for _, var in vars_and_overs]
        return best_var, useful_string_vars

    def label_changed(self):
        self.set_network_nodes()
        self.send_output()

    def send_output(self):
        if self.network is None:
            self.Outputs.network.send(None)
            self.Outputs.items.send(None)
        else:
            self.Outputs.network.send(self.network)
            self.Outputs.items.send(self.network.nodes)

    def set_network_nodes(self):
        self.Error.mismatched_lengths.clear()
        self.Information.auto_annotation.clear()
        self.Information.suggest_annotation.clear()
        if self.network is None:
            return

        data = self.data if self.data is not None else self.auto_data
        if data is None:
            self.Information.suggest_annotation()
        elif self.label_variable is None \
                and len(data) != self.network.number_of_nodes():
            self.Error.mismatched_lengths()
            data = None

        if data is None:
            self.network.nodes = self._label_to_tabel()
        elif self.label_variable is None:
            self.network.nodes = self._combined_data(data)
        else:
            self.network.nodes = self._data_by_labels(data)

    def _data_by_labels(self, data):
            data_col, _ = data.get_column_view(self.label_variable)
            data_rows = {label: row for row, label in enumerate(data_col)}
            indices = [data_rows[label] for label in self.original_nodes]
            return data[indices]

    def _combined_data(self, source):
        nodes = np.array(self.original_nodes, dtype=str)
        if nodes.ndim != 1:
            return source
        try:
            nums = np.sort(np.array([int(x) for x in nodes]))
        except ValueError:
            pass
        else:
            if np.all(nums[1:] - nums[:-1] == 1):
                return source

        src_dom = source.domain
        label_attr = StringVariable(get_unique_names(src_dom, "node_label"))
        domain = Domain(src_dom.attributes, src_dom.class_vars,
                        src_dom.metas + (label_attr, ))
        data = source.transform(domain)
        with data.unlocked(data.metas):
            data.metas[:, -1] = nodes
        return data

    def _label_to_tabel(self):
        domain = Domain([], [], [StringVariable("node_label")])
        n = len(self.original_nodes)
        data = Table.from_numpy(
            domain, np.empty((n, 0)), np.empty((n, 0)),
            np.array(self.original_nodes, dtype=str).reshape(-1, 1))
        return data


    def sendReport(self):
        self.reportSettings(
            "Network file",
            [("File name", self.filecombo.currentText()),
             ("Vertices", self.network.number_of_nodes()),
             ("Directed", gui.YesNo[self.network.edges[0].directed])
             ])


if __name__ == "__main__":
    WidgetPreview(OWNxFile).run()
