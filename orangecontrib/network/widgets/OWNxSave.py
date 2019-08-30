from Orange.widgets.widget import Input, Msg

from Orange.widgets.data.owsave import OWSaveBase
from orangecontrib.network.network import readwrite
from orangecontrib.network.network.base import Network
from orangecontrib.network.network.readwrite import PajekReader
from orangewidget.utils.widgetpreview import WidgetPreview


class OWNxSave(OWSaveBase):
    name = "Save Network"
    description = "Save network to an output file."
    icon = "icons/Save.svg"

    writers = [PajekReader]
    filters = {f"{w.DESCRIPTION} (*{w.EXTENSIONS[0]})": w
               for w in writers}

    class Inputs:
        network = Input("Network", Network, default=True)

    class Error(OWSaveBase.Error):
        multiple_edge_types = Msg("Can't save network with multiple edge types")

    @Inputs.network
    def set_network(self, network):
        if len(network.edges) > 1:
            self.Error.multiple_edge_types()
            return
        self.Error.multiple_edge_types.clear()
        self.data = network
        self.on_new_input()

    def update_status(self):
        if self.data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(
                str(self.data.number_of_nodes()),
                f"Network with {self.data.number_of_nodes()} nodes "
                f"and {self.data.number_of_edges(0)} edges.")

    def send_report(self):
        self.report_items((
            ("File name", self.filename or "not set"),
        ))


if __name__ == "__main__":  # pragma: no cover
    net = readwrite.read_pajek("../networks/leu_by_genesets.net")
    WidgetPreview(OWNxSave).run(net)
