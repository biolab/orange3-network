from collections import defaultdict
from itertools import chain

import numpy as np

from AnyQt.QtWidgets import QFormLayout

from Orange.data import DiscreteVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting, \
    Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib import network


class OWNxSingleMode(OWWidget):
    name = "Single Mode"
    description = "Convert multimodal graphs to single modal"
    icon = "icons/SingleMode.svg"
    priority = 7000

    want_main_area = False

    class Weighting:
        NoWeights, Connections, WeightedConnections = range(3)
        option_names = [
            "No weights",
            "Number of connections",
            "Weighted number of connections"]

    settingsHandler = DomainContextHandler(match_values=True)
    variable = ContextSetting(None)
    connect_value = ContextSetting(0)
    connector_value = ContextSetting(0)
    weighting = Setting(0)

    class Inputs:
        network = Input("Network", network.Graph)

    class Outputs:
        network = Output("Network", network.Graph)

    class Error(OWWidget.Error):
        no_data = Msg("Network has additional data.")
        no_categorical = Msg("Data has no categorical features.")
        same_values = Msg("Values for modes cannot be the same.")

    def __init__(self):
        super().__init__()
        self.network = None

        form = QFormLayout()
        form.setFieldGrowthPolicy(form.AllNonFixedFieldsGrow)
        gui.widgetBox(self.controlArea, box="Mode indicator", orientation=form)
        form.addRow("Feature:", gui.comboBox(
            None, self, "variable", model=VariableListModel(),
            callback=self.indicator_changed))
        form.addRow("Connect:", gui.comboBox(
            None, self, "connect_value",
            callback=self.connect_combo_changed))
        form.addRow("Connector:", gui.comboBox(
            None, self, "connector_value",
            callback=self.connector_combo_changed))

        gui.radioButtons(
            self.controlArea, self, "weighting", box="Edge weights",
            btnLabels=self.Weighting.option_names, callback=self.update_output)

        self.lbout = gui.widgetLabel(gui.hBox(self.controlArea, "Output"), "")
        self._update_combos()
        self._set_output_msg()

    @Inputs.network
    def set_network(self, network):
        self.closeContext()

        self.Error.clear()
        if network is not None:
            data = network.items()
            if data is None:
                network = None
                self.Error.no_data()

        self.network = network
        self._update_combos()
        if self.network is not None:
            self.openContext(data.domain)
        self.update_output()

    def indicator_changed(self):
        """Called on change of indicator variable"""
        self._update_value_combos()
        self.update_output()

    def connect_combo_changed(self):
        cb_connector = self.controls.connector_value
        if not cb_connector.isEnabled():
            self.connector_value = 2 - self.connect_value
        self.update_output()

    def connector_combo_changed(self):
        self.update_output()

    def _update_combos(self):
        """
        Update all three combos

        Set the combo for indicator variable and call the method to update
        combos for values"""
        model = self.controls.variable.model()
        if self.network is None:
            model.clear()
            self.variable = None
        else:
            domain = self.network.items().domain
            model[:] = [
                var for var in chain(domain.variables, domain.metas)
                if isinstance(var, DiscreteVariable) and len(var.values) >= 2]
            if not model.rowCount():
                self.Error.no_categorical()
                self.network = None
                self.variable = None
            else:
                self.variable = model[0]
        self._update_value_combos()

    def _update_value_combos(self):
        """Update combos for values"""
        cb_connect = self.controls.connect_value
        cb_connector = self.controls.connector_value
        variable = self.variable

        cb_connect.clear()
        cb_connector.clear()
        cb_connector.setDisabled(variable is None or len(variable.values) == 2)
        self.connect_value = 0
        self.connector_value = 0
        if variable is not None:
            cb_connect.addItems(variable.values)
            cb_connector.addItems(["(all others)"] + variable.values)
            self.connector_value = len(variable.values) == 2 and 2

    def update_output(self):
        """Output the network on the output"""
        self.Error.same_values.clear()
        new_net = None
        if self.network is not None:
            if self.connect_value == self.connector_value - 1:
                self.Error.same_values()
            else:
                new_net = self._compute_network()
        self.Outputs.network.send(new_net)
        self._set_output_msg(new_net)

    def _compute_network(self):
        """Compute the network for the output"""
        assert self.network is not None
        filt_data, filt_edges = self._filtered_data_edges()
        edges_commons = self._edges_and_intersections(filt_edges)
        new_edges = self._weighted_edges(edges_commons, filt_edges)

        new_net = network.Graph()
        new_net.add_nodes_from(range(len(filt_data)))
        new_net.add_edges_from(new_edges)
        new_net.set_items(filt_data)
        return new_net

    def _filtered_data_edges(self):
        """
        Compute list of edges that link connect value to the connector mode.

        Returns data rows for nodes in selected mode and edges as a
        two-column matrix. The first column contains indices corresponding
        to the new data. The second column contains indices of connector nodes
        in the original graph.
        """
        data = self.network.items()
        edges = np.array(self.network.edges())
        if not len(edges):
            return Table(data.domain), np.zeros((0, 2), dtype=np.int)
        column = data.get_column_view(self.variable)[0].astype(int)
        mode_mask = column == self.connect_value
        filt_data = data[mode_mask]
        if self.connector_value:
            conn_mask = column == self.connector_value - 1
        else:
            conn_mask = column != self.connect_value
        filt_edges = np.vstack([
            edges[mode_mask[edges[:, 0]] * conn_mask[edges[:, 1]]],
            edges[conn_mask[edges[:, 0]] * mode_mask[edges[:, 1]]][:, ::-1]
        ])
        new_idxs = np.cumsum(mode_mask) - 1
        filt_edges[:, 0] = new_idxs[filt_edges[:, 0]]
        return filt_data, filt_edges

    @staticmethod
    def _edges_and_intersections(filt_edges):
        """
        Computes edges of the new network

        Args:
            filt_edges: two-column table with relevant edges, as returned
              by `_filtered_data_edges`

        Returns:
            a generator of triplets (node1, node2, set of common neighbours)
        """
        conns = defaultdict(set)
        for node, conn in filt_edges:
            conns[node].add(conn)
        conns = list(conns.items())
        edges = (
            (node2, node1, conns1 & conns2)
            for i, (node1, conns1) in enumerate(conns)
            for node2, conns2 in conns[:i]
        )
        return ((n1, n2, common) for n1, n2, common in edges if common)

    def _weighted_edges(self, edges_inter, filt_edges):
        """
        Compute weighted edges of the new network

        Args:
            edges_inter: list of triplets (node1, node2, intersection)
            filt_edges: relevant edges, as returned by `_filtered_data_edges`

        Returns:
            list of edges for networkx.Graph; either (node1, node2) if
            unweighted, or (node1, node2, {'weight': weight})
        """
        # If trying different weighting schemas is common and speed becomes an
        # issue, edges can be stored as a list, and this function can be
        # called directly when only weighting is changed
        if self.weighting == self.Weighting.Connections:
            return [(node1, node2, {"weight": float(len(common))})
                    for node1, node2, common in edges_inter]
        elif self.weighting == self.Weighting.WeightedConnections:
            counts = np.bincount(filt_edges[:, 1]).astype(float)
            counts[counts == 0] = 1  # Prevent warnings from numpy
            weights = 1 / counts ** 2
            return [(node1, node2,
                     {"weight": sum(weights[node] for node in common)})
                    for node1, node2, common in edges_inter]
        else:  # no weights
            return [e[:2] for e in edges_inter]

    def _set_output_msg(self, out_network=None):
        if out_network is None:
            self.lbout.setText("No network on output")
        else:
            self.lbout.setText(
                f"{out_network.number_of_nodes()} nodes, "
                f"{out_network.number_of_edges()} edges")

    def send_report(self):
        if self.network:
            self.report_items("", [
                ('Input network',
                 "{} nodes, {} edges".format(
                     self.network.number_of_nodes(),
                     self.network.number_of_edges())),
                ('Mode',
                 self.variable and bool(self.variable.values) and (
                     "Select {}={}, connected through {}".format(
                         self.variable.name,
                         self.variable.values[self.connect_value],
                         "any node" if not self.connector_value
                         else self.variable.values[self.connector_value - 1]
                     ))),
                ('Weighting',
                 bool(self.weighting)
                 and self.Weighting.option_names[self.weighting]),
                ("Output network", self.lbout.text())
            ])


def main():  # pragma: no cover
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


if __name__ == "__main__":  # pragma: no cover
    main()
