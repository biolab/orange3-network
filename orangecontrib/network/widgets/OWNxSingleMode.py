from collections import defaultdict

import numpy as np

from AnyQt.QtWidgets import QFormLayout

from Orange.data import DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting, \
    Setting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib import network


class OWNxSingleMode(OWWidget):
    name = "Single Mode"
    description = "Convert multimodal graphs to single modal"
    icon = "icons/NetworkFile.svg"
    priority = 7000

    want_main_area = False

    class Weighting:
        NoWeights, Connections, WeightedConnections = range(3)
        option_names = [
            "No weights",
            "Number of connections",
            "Weighted number of connections"]

    settingsHandler = DomainContextHandler(match_values=True)
    mode_feature = ContextSetting(None)
    kept_mode = ContextSetting(0)
    connecting_mode = ContextSetting(0)
    weighting = Setting(0)

    class Inputs:
        network = Input("Network", network.Graph)

    class Outputs:
        network = Output("Network", network.Graph)

    class Error(OWWidget.Error):
        no_data = Msg("Network nodes contain no additional data.")

    def __init__(self):
        super().__init__()
        self.network = None
        self.n_output_nodes = self.n_output_edges = ""

        form = QFormLayout()
        form.setFieldGrowthPolicy(form.AllNonFixedFieldsGrow)
        gui.widgetBox(self.controlArea, box="Mode indicator", orientation=form)
        form.addRow("Feature:", gui.comboBox(
            None, self, "mode_feature",
            model=DomainModel(valid_types=DiscreteVariable),
            callback=self.indicator_changed))
        form.addRow("Selected:", gui.comboBox(
            None, self, "kept_mode", callback=self.update))
        form.addRow("Connecting:", gui.comboBox(
            None, self, "connecting_mode", callback=self.update))

        gui.radioButtons(
            self.controlArea, self, "weighting", box="Edge weights",
            btnLabels=self.Weighting.option_names, callback=self.update)

        box = gui.vBox(self.controlArea, box="Output")
        gui.label(
            box, self, "%(n_output_nodes)s nodes, %(n_output_edges)s edges")

    @Inputs.network
    def set_network(self, network):
        self.closeContext()

        self.Error.no_data.clear()
        if network is not None:
            data = network and network.items()
            if data is None:
                network = None
                self.Error.no_data()

        self.network = network
        self._update_combos()
        if network is not None:
            self.openContext(data.domain)
        self.update()

    def indicator_changed(self):
        self._update_value_combos()
        self.update()

    def _update_combos(self):
        model = self.controls.mode_feature.model()
        if self.network is None:
            model.set_domain(None)
        else:
            model.set_domain(self.network.items().domain)
        self.mode_feature = model[0] if model.rowCount() else None
        self._update_value_combos()

    def _update_value_combos(self):
        cb_kept = self.controls.kept_mode
        cb_connecting = self.controls.connecting_mode

        cb_kept.clear()
        cb_connecting.clear()
        if self.mode_feature is not None:
            cb_kept.addItems(self.mode_feature.values or ["(no values)"])
            cb_connecting.addItems(["All"] + self.mode_feature.values)
        else:
            cb_kept.addItem("(no values)")
            cb_connecting.addItem("(no values)")
        self.kept_mode = 0
        self.connecting_mode = 0

    def update(self):
        if self.network is None:
            self.Outputs.network.send(None)
            self.n_output_nodes = self.n_intermediate = self.n_output_edges = ""
            return

        data = self.network.items()
        column = data.get_column_view(self.mode_feature)[0].astype(int)
        edges = np.array(self.network.edges())
        mode_mask = column == self.kept_mode
        if self.connecting_mode:
            conn_mask = column == self.connecting_mode - 1
            edges = np.vstack([
                edges[mode_mask[edges[:, 0]] * conn_mask[edges[:, 1]]],
                edges[conn_mask[edges[:, 0]] * mode_mask[edges[:, 1]]][:, ::-1]
            ])
        else:
            edges = np.vstack([
                edges[mode_mask[edges[:, 0]]],
                edges[mode_mask[edges[:, 1]]][:, ::-1]
            ])
        new_idxs = np.cumsum(mode_mask) - 1
        edges[:, 0] = new_idxs[edges[:, 0]]
        conns = defaultdict(set)
        for node, conn in edges:
            conns[node].add(conn)
        new_edges = (
            (node1, node2, conns1 & conns2)
            for node1, conns1 in conns.items()
            for node2, conns2 in conns.items()
            if node1 < node2 and conns1 & conns2
        )
        if self.weighting == self.Weighting.NoWeights:
            new_edges = [e[:2] for e in new_edges]
        elif self.weighting == self.Weighting.Connections:
            new_edges = [(node1, node2, {"weight": float(len(common))})
                         for node1, node2, common in new_edges]
        else:  # self.weighting == self.Weighting.WeightedConnections
            counts = np.bincount(edges[:, 1]).astype(float)
            counts[counts == 0] = 1  # Just to prevent warnings
            weights = 1 / counts ** 2
            new_edges = [(node1, node2,
                          {"weight": sum(weights[node] for node in common)})
                         for node1, node2, common in new_edges]

        new_data = data[mode_mask]
        graph = network.Graph()
        graph.add_nodes_from(range(len(new_data)))
        graph.add_edges_from(new_edges)
        graph.set_items(new_data)
        self.Outputs.network.send(graph)
        self.n_output_nodes = len(new_data)
        self.n_output_edges = len(new_edges)

    def send_report(self):
        if self.network:
            self.report_items("", [
                ('Input network',
                 "{} nodes, {} edges".format(
                     self.network.number_of_nodes(),
                     self.network.number_of_edges())),
                ('Mode',
                 self.mode_feature and bool(self.mode_feature.values) and (
                     "Select {}={}, connected through {}".format(
                         self.mode_feature.name,
                         self.mode_feature.values[self.kept_mode],
                         "any node" if not self.connecting_mode
                         else self.mode_feature.values[self.connecting_mode - 1]
                     ))),
                ('Weighting',
                 bool(self.weighting)
                 and self.Weighting.option_names[self.weighting]),
                ("Output network", "{} nodes, {} edges".format(
                    self.n_output_nodes, self.n_output_edges))
            ])


def main():
    import OWNxFile
    from AnyQt.QtWidgets import QApplication
    a = QApplication([])
    ow = OWNxSingleMode()
    ow.show()

    def set_network(data, id=None):
        ow.set_network(data)

    owFile = OWNxFile.OWNxFile()
    owFile.Outputs.network.send = set_network
    owFile.openNetFile("/Users/janez/Downloads/littlenet_weighted.net")
    ow.handleNewSignals()
    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()


if __name__ == "__main__":
    main()
