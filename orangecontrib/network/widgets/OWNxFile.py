from os import path
from itertools import product
from pkg_resources import load_entry_point
from traceback import format_exception_only

import numpy as np

from AnyQt.QtWidgets import QStyle, QSizePolicy, QFileDialog

from Orange.data import Table, Domain, StringVariable
from Orange.data.util import get_unique_names
from Orange.widgets import gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from orangecontrib.network.network import Network
from orangecontrib.network.network.readwrite import read_pajek


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

    recentFiles = settings.Setting([])

    class Information(OWWidget.Information):
        auto_annotation = Msg(
            'Nodes annotated with data from file with the same name')
        suggest_annotation = Msg(
            'Add optional data input to annotate nodes')

    class Warning(OWWidget.Warning):
        auto_mismatched_lengths = Msg(
            'Data file with the same name is not used.\n'
            'The number of instances does not match the number of nodes.')
        mismatched_lengths = Msg(
            'Data size does not match the number of nodes.')

    class Error(OWWidget.Error):
        io_error = Msg('Error reading file "{}"\n{}')
        error_parsing_file = Msg('Error reading file "{}"')
        auto_data_failed = Msg(
            "Attempt to read {} failed\n"
            "The widget tried to annotated nodes with data from\n"
            "a file with the same name.")

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.network = None
        self.auto_data = None
        self.original_nodes = None
        self.data = None
        self.net_index = 0

        hb = gui.widgetBox(self.controlArea, orientation="horizontal")
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

        self.populate_comboboxes()
        self.reload()

    @Inputs.items
    def set_data(self, data):
        self.data = data
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
            startfile = next(load_entry_point(
                "Orange3-Network",
                "orange.data.io.search_paths",
                "network")())[1]
        else:
            startfile = self.recentFiles[0] if self.recentFiles else '.'

        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open a Network File', startfile,
            ';;'.join(("Pajek files (*.net *.pajek)",)))
        if not filename:
            return

        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        self.recentFiles.insert(0, filename)

        self.populate_comboboxes()
        self.net_index = 0
        self.select_net_file()

    def reload(self):
        if self.recentFiles:
            self.select_net_file()

    def select_net_file(self):
        """user selected a graph file from the combo box"""
        if self.net_index > len(self.recentFiles) - 1:
            self.browse_net_file(True)
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

    def send_output(self):
        self.set_network_nodes()

        if self.network is None:
            self.Outputs.network.send(None)
            self.Outputs.items.send(None)
            self.info.set_output_summary(self.info.NoOutput)
            return

        self.Outputs.network.send(self.network)
        self.Outputs.items.send(self.network.nodes)

        n_nodes = self.network.number_of_nodes()
        n_edges = self.network.number_of_edges()
        summary = f"{n_nodes} / {n_edges}"
        details = \
            ('Directed' if self.network.edges[0].directed else 'Undirected') \
            + f" network with\n{n_nodes} nodes and {n_edges} edges."
        self.info.set_output_summary(summary, details)

    def set_network_nodes(self):
        self.Warning.mismatched_lengths.clear()
        self.Warning.auto_mismatched_lengths.clear()
        self.Information.auto_annotation.clear()
        self.Information.suggest_annotation.clear()
        if self.network is None:
            return

        self.network.nodes = self.original_nodes
        for data_source, warning_msg, info_msg in (
            (self.data, self.Warning.mismatched_lengths, None),
            (self.auto_data, self.Warning.auto_mismatched_lengths,
             self.Information.auto_annotation)):
            if data_source is not None:
                if len(data_source) != self.network.number_of_nodes():
                    warning_msg()
                else:
                    if info_msg is not None:
                        info_msg()
                    self.network.nodes = self._combined_data(data_source)
                break
        else:
            self.network.nodes = self._label_to_tabel()
            self.Information.suggest_annotation()

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
