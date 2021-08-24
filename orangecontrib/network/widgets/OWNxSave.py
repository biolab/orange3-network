from Orange.data import StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Msg

from Orange.widgets.data.owsave import OWSaveBase
from orangecontrib.network.network import readwrite
from orangecontrib.network.network.base import Network
from orangecontrib.network.network.readwrite import PajekReader
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

    settingsHandler = DomainContextHandler()
    label_variable = ContextSetting(None)

    def __init__(self):
        super().__init__(2)

        self.label_model = DomainModel(
            placeholder="(None)", valid_types=(StringVariable, ))
        box = gui.hBox(None)
        gui.widgetLabel(box, "Node label: ")
        gui.comboBox(
            box, self, "label_variable",
            tooltip="Choose the variables that will be used as a label",
            model=self.label_model),

        self.grid.addWidget(box, 0, 0, 1, 2)
        self.grid.setRowMinimumHeight(1, 8)
        self.adjustSize()

    @Inputs.network
    def set_network(self, network):
        self.closeContext()

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
            self.openContext(domain)
        else:
            self.label_model.set_domain(None)
            self.label_variable = None
            self.controls.label_variable.setEnabled(False)

        self.on_new_input()

    def save_file(self):
        if not self.filename:
            self.save_file_as()
            return

        self.Error.general_error.clear()
        if self.data is None or not self.filename or self.writer is None:
            return
        try:
            net = self.data
            if self.label_variable is not None:
                labels = net.nodes.get_column_view(self.label_variable)[0]
            else:
                labels = range(1, net.number_of_nodes() + 1)
            self.writer.write(self.filename, net, labels)
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
