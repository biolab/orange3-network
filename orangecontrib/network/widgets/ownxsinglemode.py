from itertools import chain

from AnyQt.QtWidgets import QFormLayout

from Orange.data import DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting, \
    Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib import network
from orangecontrib.network import twomode


class OWNxSingleMode(OWWidget):
    name = "Single Mode"
    description = "Convert multimodal graphs to single modal"
    icon = "icons/SingleMode.svg"
    priority = 7000

    want_main_area = False

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
        form.addRow("by:", gui.comboBox(
            None, self, "connector_value",
            callback=self.connector_combo_changed))

        gui.comboBox(
            self.controlArea, self, "weighting", box="Edge weights",
            items=[x.name for x in twomode.Weighting],
            callback=self.update_output)

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
                mode_mask, conn_mask = self._mode_masks()
                new_net = twomode.to_single_mode(
                    self.network, mode_mask, conn_mask, self.weighting)
                new_net.set_items(self.network.items()[mode_mask])

        self.Outputs.network.send(new_net)
        self._set_output_msg(new_net)

    def _mode_masks(self):
        """Return indices of nodes in the two modes"""
        data = self.network.items()
        column = data.get_column_view(self.variable)[0].astype(int)
        mode_mask = column == self.connect_value
        if self.connector_value:
            conn_mask = column == self.connector_value - 1
        else:
            conn_mask = column != self.connect_value
        return mode_mask, conn_mask

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
                 and twomode.Weighting[self.weighting].name),
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
