import numpy as np

from Orange.data import DiscreteVariable, Table, Domain
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Output, OWWidget, Msg
from orangecontrib.network import Graph


class OWNxGroups(OWWidget):
    name = "Network Of Groups"
    description = "Group instances by feature and connect related groups."
    icon = "icons/NetworkGroups.svg"
    priority = 6435

    class Inputs:
        network = Input("Network", Graph, default=True)
        data = Input("Data", Table)

    class Outputs:
        network = Output("Network", Graph, default=True)
        data = Output("Data", Table)

    class Warning(OWWidget.Warning):
        no_graph_found = Msg("Data is given, network is missing.")
        no_discrete_features = Msg("Data has no discrete features.")

    class Error(OWWidget.Error):
        data_size_mismatch = Msg("Length of the data does not "
                                 "match the number of nodes.")

    resizing_enabled = False
    want_main_area = False

    settingsHandler = DomainContextHandler()
    feature = ContextSetting(None)

    def __init__(self):
        super().__init__()
        self.network = None
        self.data = None
        self.effective_data = None
        self.output_network = None

        info_box = gui.widgetBox(self.controlArea, "Info")
        self.input_label = gui.widgetLabel(info_box, "")
        self.__set_input_label_text()

        feature_box = gui.widgetBox(self.controlArea, "Group by")
        self.feature_model = DomainModel(valid_types=DiscreteVariable)
        self.feature_combo = gui.comboBox(
            feature_box, self, "feature", contentsLength=15,
            callback=self.__feature_combo_changed, model=self.feature_model
        )

    def __set_input_label_text(self):
        text = "No data on input."
        if self.network is not None:
            dir_ = "Directed" if self.network.is_directed() else "Undirected"
            text = f"{dir_} graph\n{self.network.number_of_nodes()}" \
                   f" nodes, {self.network.number_of_edges()} edges"
        self.input_label.setText(text)

    def __feature_combo_changed(self):
        self.commit()

    @Inputs.network
    def set_network(self, network):
        self.network = network
        self.__set_input_label_text()

    @Inputs.data
    def set_data(self, data):
        self.data = data

    def handleNewSignals(self):
        self.closeContext()
        self.clear_messages()
        self.set_effective_data()
        self.set_feature_model()
        if self.feature_model:
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
            self.effective_data = self.network.items()

        if self.effective_data is not None and not \
                self.effective_data.domain.has_discrete_attributes(True, True):
            self.Warning.no_discrete_features()

    def set_feature_model(self):
        data = self.effective_data
        self.feature_model.set_domain(data and data.domain)
        self.feature = self.feature_model[0] if self.feature_model else None

    def commit(self):
        if self.feature is None:
            self.output_network = None
            self.Outputs.network.send(None)
            self.Outputs.data.send(None)
            return

        self._map_network()
        self.Outputs.network.send(self.output_network)
        self.Outputs.data.send(self.output_network.items())

    def _map_network(self):
        edges = self.network.edges(data='weight')
        row, col, weights = zip(*edges)
        row, col = self._map_into_feature_values(np.array(row), np.array(col))
        edges = self._construct_edges(row, col, np.array(weights))

        network = Graph()
        network.add_nodes_from(range(len(self.feature.values)))
        network.add_weighted_edges_from(edges)
        network.set_items(self._construct_items())
        self.output_network = network

    def _map_into_feature_values(self, row, col):
        selected_column = self.effective_data.get_column_view(self.feature)[0]
        return (selected_column[row].astype(np.float64),
                selected_column[col].astype(np.float64))

    @staticmethod
    def _construct_edges(col, row, weights):
        # remove edges that connect to "unknown" group
        mask = ~np.any(np.isnan(np.vstack((row, col))), axis=0)
        # remove edges within a node
        mask = np.logical_and((row != col), mask)
        row, col, weights = row[mask], col[mask], weights[mask]

        # find unique edges
        mask = row > col
        row[mask], col[mask] = col[mask], row[mask]

        array = np.vstack((row, col))
        (row, col), inverse = np.unique(array, axis=1, return_inverse=True)

        # assign each edge the sum of weights of belonging original edges
        weights = np.array(
            [np.sum(weights[inverse == i]) for i in range(len(row))])
        return [(int(u), int(v), w) for u, v, w in zip(row, col, weights)]

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
        self.report_items("Group by", [("Feature", self.feature.name)])
        if self.output_network:
            self.report_items("Output network", [
                ("Number of vertices", self.output_network.number_of_nodes()),
                ("Number of edges", self.output_network.number_of_edges())])
            self.report_data("Output data", self.output_network.items())


def main():
    from os.path import join, dirname
    from AnyQt.QtWidgets import QApplication
    from orangecontrib.network.widgets.OWNxFile import OWNxFile

    app = QApplication([])
    ow = OWNxGroups()
    ow.show()

    def set_network(data):
        ow.set_network(data)

    ow_file = OWNxFile()
    ow_file.Outputs.network.send = set_network
    ow_file.openNetFile(join(dirname(dirname(__file__)),
                             "networks", "airtraffic.net"))
    ow.handleNewSignals()
    app.exec_()
    ow.saveSettings()


if __name__ == "__main__":
    main()
