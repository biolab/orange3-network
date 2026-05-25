from Orange.data import StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Msg

from Orange.widgets.data.owsave import OWSaveBase
from orangecontrib.network.network import readwrite
from orangecontrib.network.network.base import Network
from orangecontrib.network.network.readwrite import PajekReader, \
    dict_rows_from_table
from orangewidget.settings import Setting
from orangewidget.utils.widgetpreview import WidgetPreview


class OWNxSave(OWSaveBase):
    name = "Save Network"
    description = "Save network to an output file."
    icon = "icons/NetworkSave.svg"

    writers = [PajekReader]
    filters = {f"{w.DESCRIPTION} (*{w.EXTENSIONS[0]})": w
               for w in writers}

    class Inputs:
        network = Input("Network", Network, default=True)

    class Error(OWSaveBase.Error):
        multiple_edge_types = Msg("Can't save network with multiple edge types")

    label_variable_hint = Setting(None)
    edge_variable_hint = Setting(None)
    append_node_data = Setting(False)
    append_edge_data = Setting(False)

    def __init__(self):
        super().__init__(2)
        self.label_variable = None
        self.edge_variable = None

        self.grid.setSpacing(24)

        box = gui.vBox(None, "Node Labels")
        self.label_model = DomainModel(
            placeholder="(None)", valid_types=(StringVariable, ))
        gui.comboBox(
            box, self, "label_variable",
            tooltip="Choose the variable that will be used as a node label",
            model=self.label_model,
            callback=self.update_label_hint)
        gui.checkBox(
            box, self, "append_node_data",
            "Append other available node data"
        )
        self.grid.addWidget(box, 0, 0, 1, 2)

        box = gui.vBox(None, "Edge Labels")
        self.edge_model = DomainModel(
            placeholder="(None)", valid_types=(StringVariable, ))
        gui.comboBox(
            box, self, "edge_variable",
            tooltip="Choose the variable that will be used as an edge label",
            model=self.edge_model,
            callback=self.update_edge_hint)
        gui.checkBox(
            box, self, "append_edge_data",
            "Append other available edge data"
        )
        self.grid.addWidget(box, 1, 0, 1, 2)

        self.adjustSize()

    def update_label_hint(self):
        if self.label_variable is not None:
            self.label_variable_hint = self.label_variable.name
        else:
            self.label_variable_hint = None

    def update_edge_hint(self):
        if self.edge_variable is not None:
            self.edge_variable_hint = self.edge_variable.name
        else:
            self.edge_variable_hint = None

    @Inputs.network
    def set_network(self, network):
        self.data = network
        if network is None:
            return
        if len(network.edges) > 1:
            self.Error.multiple_edge_types()
            self.data = None
            return
        self.Error.multiple_edge_types.clear()

        if isinstance(network.nodes, Table):
            self.controls.label_variable.setEnabled(True)
            domain = network.nodes.domain
            self.label_model.set_domain(domain)
            for attr in domain.metas:
                if attr.name == "node_label":
                    self.label_variable = attr
                    break
            if self.label_variable_hint in domain and \
                    (attr := domain[self.label_variable_hint]) in self.label_model:
                self.label_variable = domain[self.label_variable_hint]
        else:
            self.label_model.set_domain(None)
            self.label_variable = None
            self.controls.label_variable.setEnabled(False)

        if isinstance(network.edges[0].edge_data, Table):
            self.controls.edge_variable.setEnabled(True)
            domain = network.edges.domain
            self.edge_model.set_domain(domain)
            for attr in domain.metas:
                if attr.name == "edge_label":
                    self.edge_variable = attr
                    break
            if self.edge_variable_hint in domain and \
                    (attr := domain[self.edge_variable]) in self.edge_model:
                self.edge_variable = attr
        else:
            self.edge_model.set_domain(None)
            self.edge_variable = None
            self.controls.edge_variable.setEnabled(False)

        self.on_new_input()

    def do_save(self):
        if not self.filename:
            self.save_file_as()
            return

        self.Error.general_error.clear()
        if self.data is None or not self.filename or self.writer is None:
            return
        try:
            net = self.data
            if self.label_variable is not None:
                labels = net.nodes.get_column(self.label_variable)
            else:
                labels = range(1, net.number_of_nodes() + 1)
            if self.append_node_data and isinstance(data := net.nodes, Table):
                label_data = dict_rows_from_table(data, self.label_variable)
            else:
                label_data = None
            if self.append_edge_data and isinstance(data := net.edges[0].edge_data, Table):
                edge_data = dict_rows_from_table(data, self.edge_variable)
            else:
                edge_data = None
            self.writer.write(self.filename, net, labels, label_data, edge_data)
        except IOError as err_value:
            self.Error.general_error(str(err_value))

    def send_report(self):
        self.report_items((
            ("Node labels",
             self.label_variable.name if self.label_variable else "none"),
            ("File name", self.filename or "not set"),
        ))


def main_with_annotation():
    from AnyQt.QtWidgets import QApplication
    from OWNxFile import OWNxFile
    app = QApplication([])
    file_widget = OWNxFile()
    file_widget.Outputs.network.send = WidgetPreview(OWNxSave).run
    file_widget.open_net_file("../networks/leu_by_genesets.net")


def main_without_annotation():
    net = readwrite.read_pajek("../networks/leu_by_genesets.net")
    WidgetPreview(OWNxSave).run(net)


if __name__ == "__main__":  # pragma: no cover
    main_with_annotation()
