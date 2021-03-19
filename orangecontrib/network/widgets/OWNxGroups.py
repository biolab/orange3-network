import numpy as np
import scipy.sparse as sp

from Orange.data import DiscreteVariable, Table, Domain
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, \
    Setting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Output, OWWidget, Msg
from orangecontrib.network import Network


class OWNxGroups(OWWidget):
    name = "Network Of Groups"
    description = "Group instances by feature and connect related groups."
    icon = "icons/NetworkGroups.svg"
    priority = 6435

    class Inputs:
        network = Input("Network", Network, default=True)
        data = Input("Data", Table)

    class Outputs:
        network = Output("Network", Network, default=True)
        data = Output("Data", Table)

    class Warning(OWWidget.Warning):
        no_graph_found = Msg("Data is given, network is missing.")
        no_discrete_features = Msg("Data has no discrete features.")

    class Error(OWWidget.Error):
        data_size_mismatch = Msg("Length of the data does not "
                                 "match the number of nodes.")

    resizing_enabled = False
    want_main_area = False

    NoWeights, WeightByDegrees, WeightByWeights = range(3)
    weight_labels =\
        ["No weights", "Number of connections", "Sum of connection weights"]

    settingsHandler = DomainContextHandler()
    feature = ContextSetting(None)
    weighting = Setting(2)
    normalize = Setting(True)

    def __init__(self):
        super().__init__()
        self.network = None
        self.data = None
        self.effective_data = None
        self.out_nodes = self.out_edges = None

        info_box = gui.widgetBox(self.controlArea, "Info")
        self.input_label = gui.widgetLabel(info_box, "")
        self.output_label = gui.widgetLabel(info_box, "")
        self._set_input_label_text()
        self._set_output_label_text(None)

        gui.comboBox(
            self.controlArea, self, "feature", box="Group by",
            callback=self.__feature_combo_changed,
            model=DomainModel(valid_types=DiscreteVariable)
        )
        radios = gui.radioButtons(
            self.controlArea, self, "weighting", box="Output weights",
            btnLabels=self.weight_labels, callback=self.__feature_combo_changed
        )
        gui.checkBox(
            gui.indentedBox(radios),
            self, "normalize", "Normalize by geometric mean",
            callback=self.__normalization_changed
        )
        self.controls.normalize.setEnabled(
            self.weighting == self.WeightByWeights)

    def _set_input_label_text(self):
        if self.network is None:
            self.input_label.setText("Input: no data")
        else:
            self.input_label.setText(
                f"Input: "
                f"{self.network.number_of_nodes()} nodes, "
                f"{self.network.number_of_edges()} edges")

    def _set_output_label_text(self, output_network):
        if output_network is None:
            self.output_label.setText("Output: no data")
            self.out_nodes = self.out_edges = None
        else:
            self.out_nodes = output_network.number_of_nodes()
            self.out_edges = output_network.number_of_edges()
            self.output_label.setText(
                f"Output: {self.out_nodes} nodes, {self.out_edges} edges"
            )

    def __feature_combo_changed(self):
        self.controls.normalize.setEnabled(
            self.weighting == self.WeightByWeights)
        self.commit()

    def __normalization_changed(self):
        self.commit()

    @Inputs.network
    def set_network(self, network):
        self.network = network
        self._set_input_label_text()

    @Inputs.data
    def set_data(self, data):
        self.data = data

    def handleNewSignals(self):
        self.closeContext()
        self.clear_messages()
        self.set_effective_data()
        self.set_feature_model()
        if self.controls.feature.model():
            self.openContext(self.effective_data)
        self.commit()

    def set_effective_data(self):
        self.effective_data = None
        if self.network is None and self.data is not None:
            self.Warning.no_graph_found()
        elif self.network is not None and self.data is not None:
            if len(self.data) != self.network.number_of_nodes():
                self.Error.data_size_mismatch()
            else:
                self.effective_data = self.data
        elif self.data is None and self.network is not None:
            self.effective_data = self.network.nodes

        if self.effective_data is not None and not \
                self.effective_data.domain.has_discrete_attributes(True, True):
            self.Warning.no_discrete_features()

    def set_feature_model(self):
        data = self.effective_data
        feature_model = self.controls.feature.model()
        feature_model.set_domain(data.domain if data else None)
        self.feature = feature_model[0] if feature_model else None

    def commit(self):
        if self.feature is None:
            output_network = None
        else:
            output_network = self._map_network()
        self.Outputs.network.send(output_network)
        self.Outputs.data.send(output_network and output_network.nodes)
        self._set_output_label_text(output_network)

    def _map_network(self):
        edges = self.network.edges[0].edges.tocoo()
        row, col = edges.row, edges.col
        if self.weighting == self.WeightByWeights:
            weights = edges.data
        else:
            weights = None
        if self.normalize:
            self._normalize_weights(row, col, weights)
        row, col = self._map_into_feature_values(row, col)
        return Network(
            self._construct_items(),
            self._construct_edges(row, col, weights))

    def _normalize_weights(self, row, col, weights):
        if weights is None:
            weights = np.ones((len(row)), dtype=float)
            degs = self.network.degrees()
        else:
            degs = self.network.degrees(weighted=True)
        weights /= np.sqrt(degs.T[row] * degs.T[col])

    def _map_into_feature_values(self, row, col):
        selected_column = self.effective_data.get_column_view(self.feature)[0]
        return (selected_column[row].astype(np.float64),
                selected_column[col].astype(np.float64))

    def _construct_edges(self, col, row, weights):
        # remove edges that connect to "unknown" group
        mask = ~np.any(np.isnan(np.vstack((row, col))), axis=0)
        # remove edges within a node
        mask = np.logical_and((row != col), mask)
        row, col = row[mask], col[mask]
        if weights is not None:
            weights = weights[mask]

        # find unique edges
        mask = row > col
        row[mask], col[mask] = col[mask], row[mask]

        array = np.vstack((row.astype(int), col.astype(int)))
        (row, col), inverse = np.unique(array, axis=1, return_inverse=True)

        if self.weighting == self.NoWeights:
            data = np.ones(len(row))
        elif self.weighting == self.WeightByDegrees:
            data = np.fromiter(
                (np.sum(inverse == i).astype(float) for i in range(len(row))),
                dtype=float, count=len(row))
        else:
            data = np.fromiter(
                (np.sum(weights[inverse == i]) for i in range(len(row))),
                dtype=float, count=len(row))
        dim = len(self.feature.values)
        return sp.csr_matrix((data, (row, col)), shape=(dim, dim))

    def _construct_items(self):
        domain = Domain([self.feature])
        return Table(domain, np.arange(len(self.feature.values))[:, None])

    def send_report(self):
        if not self.effective_data:
            return

        self.report_items("Input network", [
            ("Number of vertices", self.network.number_of_nodes()),
            ("Number of edges", self.network.number_of_edges())])
        self.report_data("Input data", self.effective_data)
        self.report_items("Settings", [
            ("Group by", self.feature.name),
            ("Weights", self.weight_labels[self.weighting].lower() +
             (", normalized by geometric mean" if self.normalize else ""))
        ])
        if self.out_nodes is not None:
            self.report_items("Output network", [
                ("Number of vertices", self.out_nodes),
                ("Number of edges", self.out_edges)])


def main():  # pragma: no cover
    from orangecontrib.network.network.readwrite import read_pajek
    from os.path import join, dirname
    from orangewidget.utils.widgetpreview import WidgetPreview

    path = join(dirname(__file__), "..", "networks")
    network = read_pajek(join(path, 'airtraffic.net'))
    data = Table(join(path, 'airtraffic_items.tab'))

    WidgetPreview(OWNxGroups).run(set_network=network, set_data=data)


if __name__ == "__main__":  # pragma: no cover
    main()
