import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QTimer, QSize, Qt, Signal, QObject, QThread

import Orange
from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting, SettingProvider
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.widget import Input, Output

from orangecontrib.network.network.base import Network
from orangecontrib.network.network.layout import fruchterman_reingold
from orangecontrib.network.widgets.graphview import GraphView

FR_ALLOWED_TIME = 30


class OWNxExplorer(OWDataProjectionWidget):
    name = "Network Explorer"
    description = "Visually explore the network and its properties."
    icon = "icons/NetworkExplorer.svg"
    priority = 6420

    class Inputs:
        node_data = Input("Node Data", Table)
        node_subset = Input("Node Subset", Table)
        network = Input("Network", Network, default=True)
        node_distances = Input("Node Distances", Orange.misc.DistMatrix)

    class Outputs(OWDataProjectionWidget.Outputs):
        subgraph = Output("Selected sub-network", Network)
        unselected_subgraph = Output("Remaining sub-network", Network)
        distances = Output("Distance matrix", Orange.misc.DistMatrix)

    UserAdviceMessages = [
        widget.Message("Double clicks select connected components",
                       widget.Message.Information),
    ]

    GRAPH_CLASS = GraphView
    graph = SettingProvider(GraphView)

    layout_density = Setting(10)
    observe_weights = Setting(True)

    mark_hops = Setting(1)
    mark_min_conn = Setting(5)
    mark_max_conn = Setting(5)
    mark_most_conn = Setting(1)

    alpha_value = 255  # Override the setting from parent

    class Warning(OWDataProjectionWidget.Warning):
        distance_matrix_mismatch = widget.Msg(
            "Distance matrix size doesn't match the number of network nodes "
            "and will be ignored.")
        no_graph_found = widget.Msg("Node data is given, graph data is missing.")

    class Error(OWDataProjectionWidget.Error):
        data_size_mismatch = widget.Msg(
            "Length of the data does not match the number of nodes.")
        network_too_large = widget.Msg("Network is too large to visualize.")
        single_node_graph = widget.Msg("I don't do single-node graphs today.")

    def __init__(self):
        # These are already needed in super().__init__()
        self.number_of_nodes = 0
        self.number_of_edges = 0
        self.nHighlighted = 0
        self.nSelected = 0
        self.nodes_per_edge = 0
        self.edges_per_node = 0

        self.mark_mode = 0
        self.mark_text = ""

        super().__init__()

        self.network = None
        self.node_data = None
        self.distance_matrix = None
        self.edges = None
        self.positions = None

        self._optimizer = None
        self._animation_thread = None
        self._stop_optimization = False

        self.marked_nodes = None
        self.searchStringTimer = QTimer(self)
        self.searchStringTimer.timeout.connect(self.update_marks)
        self.set_mark_mode()
        self.setMinimumWidth(600)

    def sizeHint(self):
        return QSize(800, 600)

    def _add_controls(self):
        self.gui = OWPlotGUI(self)
        self._add_info_box()
        self.gui.point_properties_box(self.controlArea)
        self._add_effects_box()
        self.gui.plot_properties_box(self.controlArea)
        self._add_mark_box()
        self.controls.attr_label.activated.connect(self.on_change_label_attr)

    def _add_info_box(self):
        info = gui.vBox(self.controlArea, box="Layout")
        gui.label(
            info, self,
            "Nodes: %(number_of_nodes)i (%(nodes_per_edge).2f per edge); "
            "%(nSelected)i selected")
        gui.label(
            info, self,
            "Edges: %(number_of_edges)i (%(edges_per_node).2f per node)")
        lbox = gui.hBox(info)
        self.relayout_button = gui.button(
            lbox, self, 'Improve', callback=self.improve, autoDefault=False,
            tooltip="Optimize the current layout, with a small initial jerk")
        self.stop_button = gui.button(
            lbox, self, 'Stop', callback=self.stop_relayout, autoDefault=False,
            hidden=True)
        self.randomize_button = gui.button(
            lbox, self, 'Re-layout', callback=self.restart, autoDefault=False,
            tooltip="Restart laying out from random positions")
        gui.hSlider(info, self, "layout_density", minValue=1, maxValue=50,
                    label="Gravity", orientation=Qt.Horizontal,
                    callback_finished=self.improve,
                    tooltip="Lower values improve optimization,\n"
                            "higher work better for graph with many small "
                            "components")
        gui.checkBox(info, self, "observe_weights",
                     label="Make edges with large weights shorter",
                     callback=self.improve)

    def _add_effects_box(self):
        gbox = self.gui.create_gridbox(self.controlArea, box="Widths and Sizes")
        self.gui.add_widget(self.gui.PointSize, gbox)
        gbox.layout().itemAtPosition(1, 0).widget().setText("Node Size:")
        self.gui.add_control(
            gbox, gui.hSlider, "Edge width:",
            master=self, value='graph.edge_width',
            minValue=1, maxValue=10, step=1,
            callback=self.graph.update_edges)
        box = gui.vBox(None)
        gbox.layout().addWidget(box, 3, 0, 1, 2)
        gui.separator(box)
        self.checkbox_relative_edges = gui.checkBox(
            box, self, 'graph.relative_edge_widths',
            'Scale edge widths to weights',
            callback=self.graph.update_edges)
        self.checkbox_show_weights = gui.checkBox(
            box, self, 'graph.show_edge_weights',
            'Show edge weights',
            callback=self.graph.update_edge_labels)
        self.checkbox_show_weights = gui.checkBox(
            box, self, 'graph.label_selected_edges',
            'Label only edges of selected nodes',
            callback=self.graph.update_edge_labels)

        # This is ugly: create a slider that controls alpha_value so that
        # parent can enable and disable it - although it's never added to any
        # layout and visible to the user
        gui.hSlider(None, self, "graph.alpha_value")

    def _add_mark_box(self):
        hbox = gui.hBox(None, box=True)
        self.mainArea.layout().addWidget(hbox)
        vbox = gui.hBox(hbox)

        def spin(value, label, minv, maxv):
            return gui.spin(
                vbox, self, value, label=label, minv=minv, maxv=maxv,
                step=1,
                alignment=Qt.AlignRight, callback=self.update_marks).box

        def text_line():
            def set_search_string_timer():
                self.searchStringTimer.stop()
                self.searchStringTimer.start(300)

            return gui.lineEdit(
                gui.hBox(vbox), self, "mark_text", label="Text: ",
                orientation=Qt.Horizontal, minimumWidth=50,
                callback=set_search_string_timer, callbackOnType=True).box

        def _mark_by_labels(marker):
            txt = self.mark_text.lower()
            if not txt:
                return None
            labels = self.get_label_data()
            if labels is None:
                return None
            return marker(np.char.array(labels), txt)

        def mark_label_starts():
            return _mark_by_labels(
                lambda labels, txt: np.flatnonzero(labels.lower().startswith(txt)))

        def mark_label_contains():
            return _mark_by_labels(
                lambda labels, txt: np.flatnonzero(labels.lower().find(txt) != -1))

        def mark_text():
            txt = self.mark_text.lower()
            if not txt or self.data is None:
                return None
            return np.array(
                [i for i, inst in enumerate(self.data)
                 if txt in "\x00".join(map(str, inst.list)).lower()])

        def mark_reachable():
            selected = self.graph.get_selection()
            if selected is None:
                return None
            return self.get_reachable(selected)

        def mark_close():
            selected = self.graph.get_selection()
            if selected is None:
                return None
            neighbours = set(selected)
            last_round = list(neighbours)
            for _ in range(self.mark_hops):
                next_round = set()
                for neigh in last_round:
                    next_round |= set(self.network.neighbours(neigh))
                neighbours |= next_round
                last_round = next_round
            neighbours -= set(selected)
            return np.array(list(neighbours))

        def mark_from_input():
            if self.subset_data is None or self.data is None:
                return None
            ids = set(self.subset_data.ids)
            return np.array(
                [i for i, ex in enumerate(self.data) if ex.id in ids])

        def mark_most_connections():
            n = self.mark_most_conn
            if n >= self.number_of_nodes:
                return np.arange(self.number_of_nodes)
            degrees = self.network.degrees()
            # pylint: disable=invalid-unary-operand-type
            min_degree = np.partition(degrees, -n)[-n]
            return np.flatnonzero(degrees >= min_degree)

        def mark_more_than_any_neighbour():
            degrees = self.network.degrees()
            return np.array(
                [node for node, degree in enumerate(degrees)
                 if degree > np.max(degrees[self.network.neighbours(node)],
                                    initial=0)])

        def mark_more_than_average_neighbour():
            degrees = self.network.degrees()
            return np.array(
                [node for node, degree, neighbours in (
                    (node, degree, self.network.neighbours(node))
                     for node, degree in enumerate(degrees))
                 if degree > (np.mean(degrees[neighbours]) if neighbours.size else 0)
                 ]
            )

        self.mark_criteria = [
            ("(Select criteria for marking)", None, lambda: np.zeros((0,))),
            ("Mark nodes whose label starts with", text_line(), mark_label_starts),
            ("Mark nodes whose label contains", text_line(), mark_label_contains),
            ("Mark nodes whose data that contains", text_line(), mark_text),
            ("Mark nodes reachable from selected", None, mark_reachable),

            ("Mark nodes in vicinity of selection",
             spin("mark_hops", "Number of hops:", 1, 20),
             mark_close),

            ("Mark nodes from subset signal", None, mark_from_input),

            ("Mark nodes with few connections",
             spin("mark_max_conn", "Max. connections:", 0, 1000),
             lambda: np.flatnonzero(self.network.degrees() <= self.mark_max_conn)),

            ("Mark nodes with many connections",
             spin("mark_min_conn", "Min. connections:", 1, 1000),
             lambda: np.flatnonzero(self.network.degrees() >= self.mark_min_conn)),

            ("Mark nodes with most connections",
             spin("mark_most_conn", "Number of marked:", 1, 1000),
             mark_most_connections),

            ("Mark nodes with more connections than any neighbour", None,
             mark_more_than_any_neighbour),

            ("Mark nodes with more connections than average neighbour", None,
             mark_more_than_average_neighbour)
        ]

        cb = gui.comboBox(
            hbox, self, "mark_mode",
            items=[item for item, *_ in self.mark_criteria],
            maximumContentsLength=-1, callback=self.set_mark_mode)
        hbox.layout().insertWidget(0, cb)

        gui.rubber(hbox)
        self.btselect = gui.button(
            hbox, self, "Select", callback=self.select_marked)
        self.btadd = gui.button(
            hbox, self, "Add to Selection", callback=self.select_add_marked)
        self.btgroup = gui.button(
            hbox, self, "Add New Group", callback=self.select_as_group)

    def set_mark_mode(self, mode=None):
        if mode is not None:
            self.mark_mode = mode
        for i, (_, widget, _) in enumerate(self.mark_criteria):
            if widget:
                if i == self.mark_mode:
                    widget.show()
                else:
                    widget.hide()
        self.searchStringTimer.stop()
        self.update_marks()

    def update_marks(self):
        if self.network is None:
            return
        self.marked_nodes = self.mark_criteria[self.mark_mode][2]()
        if self.marked_nodes is not None and not self.marked_nodes.size:
            self.marked_nodes = None
        self.graph.update_marks()
        if self.graph.label_only_selected:
            self.graph.update_labels()
        self.update_selection_buttons()

    def update_selection_buttons(self):
        if self.marked_nodes is None:
            self.btselect.hide()
            self.btadd.hide()
            self.btgroup.hide()
            return
        else:
            self.btselect.show()

        selection = self.graph.get_selection()
        if not len(selection) or np.max(selection) == 0:
            self.btadd.hide()
            self.btgroup.hide()
        elif np.max(selection) == 1:
            self.btadd.setText("Add to Selection")
            self.btadd.show()
            self.btgroup.hide()
        else:
            self.btadd.setText("Add to Group")
            self.btadd.show()
            self.btgroup.show()

    def selection_changed(self):
        super().selection_changed()
        self.nSelected = 0 if self.selection is None else len(self.selection)
        self.update_selection_buttons()
        self.update_marks()

    def select_marked(self):
        self.graph.selection_select(self.marked_nodes)

    def select_add_marked(self):
        self.graph.selection_append(self.marked_nodes)

    def select_as_group(self):
        self.graph.selection_new_group(self.marked_nodes)

    def on_change_label_attr(self):
        if self.mark_mode in (1, 2):
            self.update_marks()

    @Inputs.node_data
    def set_node_data(self, data):
        self.node_data = data

    @Inputs.node_subset
    def set_node_subset(self, data):
        # It would be better to call super, but this fails because super
        # is decorated to set the partial summary for signal "Subset Data",
        # which does not exist for this widget (OWNxExplorer.Inputs is not
        # derived from OWDataProjectionWidget.Inputs in order to rename the
        # signal)
        self.subset_data = data

    @Inputs.node_distances
    def set_items_distance_matrix(self, matrix):
        self.distance_matrix = matrix
        self.positions = None

    @Inputs.network
    def set_graph(self, graph):

        def set_graph_none(error=None):
            if error is not None:
                error()
            self.network = None
            self.number_of_nodes = self.edges_per_node = 0
            self.number_of_edges = self.nodes_per_edge = 0

        def compute_stats():
            self.number_of_nodes = graph.number_of_nodes()
            self.number_of_edges = graph.number_of_edges()
            self.edges_per_node = self.number_of_edges / self.number_of_nodes
            self.nodes_per_edge = \
                self.number_of_nodes / max(1, self.number_of_edges)

        self.mark_text = ""
        self.set_mark_mode(0)
        self.positions = None

        if not graph or graph.number_of_nodes() == 0:
            set_graph_none()
            return
        if graph.number_of_nodes() + graph.number_of_edges() > 100000:
            set_graph_none(self.Error.network_too_large)
            return
        self.Error.clear()

        self.network = graph
        compute_stats()

    def handleNewSignals(self):
        network = self.network

        def set_actual_data():
            self.closeContext()
            self.Error.data_size_mismatch.clear()
            self.Warning.no_graph_found.clear()
            self._invalid_data = False
            if network is None:
                if self.node_data is not None:
                    self.Warning.no_graph_found()
                return
            n_nodes = len(self.network.nodes)
            if self.node_data is not None:
                if len(self.node_data) != n_nodes:
                    self.Error.data_size_mismatch()
                    self._invalid_data = True
                    self.data = None
                else:
                    self.data = self.node_data
            if self.node_data is None:
                if isinstance(network.nodes, Table):
                    self.data = network.nodes
                elif isinstance(network.nodes, np.ndarray) \
                        and (len(network.nodes.shape) == 1
                             or network.nodes.shape[1] == 1):
                    self.data = Table.from_numpy(
                        Domain([], None, [StringVariable("label")]),
                        np.zeros((len(network.nodes),0)),
                        None,
                        metas=network.nodes.reshape((n_nodes, 1))
                    )
                else:
                    self.data = None

            if self.data is not None:
                # Replicate the necessary parts of set_data
                self.valid_data = np.full(len(self.data), True, dtype=bool)
                self.init_attr_values()
                self.openContext(self.data)
                self.cb_class_density.setEnabled(self.can_draw_density())

        def set_actual_edges():
            def set_checkboxes(value):
                self.checkbox_show_weights.setEnabled(value)
                self.checkbox_relative_edges.setEnabled(value)

            self.Warning.distance_matrix_mismatch.clear()

            if self.network is None:
                self.edges = None
                set_checkboxes(False)
                return

            set_checkboxes(True)
            if network.number_of_edges(0):
                self.edges = network.edges[0].edges.tocoo()
            else:
                self.edges = sp.coo_matrix((0, 3))
            if self.distance_matrix is not None:
                if len(self.distance_matrix) != self.number_of_nodes:
                    self.Warning.distance_matrix_mismatch()
                else:
                    self.edges.data = np.fromiter(
                        (self.distance_matrix[u, v]
                         for u, v in zip(self.edges.row, self.edges.col)),
                        dtype=np.int32, count=len(self.edges.row)
                    )
            if np.allclose(self.edges.data, 0):
                self.edges.data[:] = 1
                set_checkboxes(False)
            elif len(set(self.edges.data)) == 1:
                set_checkboxes(False)

        self.stop_optimization_and_wait()
        set_actual_data()
        super()._handle_subset_data()
        if self.positions is None:
            set_actual_edges()
            self.set_random_positions()
            self.graph.reset_graph()
            self.relayout(True)
        else:
            self.graph.update_point_props()
        self.update_marks()
        self.update_selection_buttons()

    def init_attr_values(self):
        super().init_attr_values()
        if self.node_data is None \
                and self.data is not None \
                and isinstance(self.network.nodes, np.ndarray):
            assert len(self.data.domain.metas) == 1
            self.attr_label = self.data.domain.metas[0]

    def randomize(self):
        self.set_random_positions()
        self.graph.update_coordinates()

    def set_random_positions(self):
        if self.network is None:
            self.position = None
        else:
            self.positions = np.random.uniform(size=(self.number_of_nodes, 2))

    def get_reachable(self, initial):
        to_check = list(initial)
        reachable = set(to_check)
        for node in to_check:
            new_checks = set(self.network.neighbours(node)) - reachable
            to_check += new_checks
            reachable |= new_checks
        return np.array(to_check)

    def send_data(self):
        super().send_data()

        Outputs = self.Outputs
        selected_indices = self.graph.get_selection()
        if selected_indices is None or len(selected_indices) == 0:
            Outputs.subgraph.send(None)
            Outputs.unselected_subgraph.send(self.network)
            Outputs.distances.send(None)
            return

        selection = self.graph.selection
        subgraph = self.network.subgraph(selected_indices)
        subgraph.nodes = \
            self._get_selected_data(self.data, selected_indices, selection)
        Outputs.subgraph.send(subgraph)
        Outputs.unselected_subgraph.send(
            self.network.subgraph(np.flatnonzero(selection == 0)))
        distances = self.distance_matrix
        if distances is None:
            Outputs.distances.send(None)
        else:
            Outputs.distances.send(distances.submatrix(sorted(selected_indices)))

    def get_coordinates_data(self):
        if self.positions is not None:
            return self.positions.T
        else:
            return None, None

    def get_embedding(self):
        return self.positions

    def get_subset_mask(self):
        if self.data is None:
            return None
        return super().get_subset_mask()

    def get_edges(self):
        return self.edges

    def is_directed(self):
        return self.network is not None and self.network.edges[0].directed

    def get_marked_nodes(self):
        return self.marked_nodes

    def set_buttons(self, running):
        self.stop_button.setHidden(not running)
        self.relayout_button.setHidden(running)
        self.randomize_button.setHidden(running)

    def stop_relayout(self):
        self._stop_optimization = True
        self.set_buttons(running=False)

    def restart(self):
        self.relayout(restart=True)

    def improve(self):
        self.relayout(restart=False)

    # TODO: Stop relayout if new data is received
    def relayout(self, restart):
        if self.edges is None:
            return
        if restart or self.positions is None:
            self.set_random_positions()
        self.progressbar = gui.ProgressBar(self, 100)
        self.set_buttons(running=True)
        self._stop_optimization = False

        Simplifications = self.graph.Simplifications
        self.graph.set_simplifications(
            Simplifications.NoDensity
            + Simplifications.NoLabels * (len(self.graph.labels) > 20)
            + Simplifications.NoEdgeLabels * (len(self.graph.edge_labels) > 20)
            + Simplifications.NoEdges * (self.number_of_edges > 30000))

        large_graph = self.number_of_nodes + self.number_of_edges > 30000

        class LayoutOptimizer(QObject):
            update = Signal(np.ndarray, float)
            done = Signal(np.ndarray)
            stopped = Signal()

            def __init__(self, widget):
                super().__init__()
                self.widget = widget

            def send_update(self, positions, progress):
                if not large_graph:
                    self.update.emit(np.array(positions), progress)
                return not self.widget._stop_optimization

            def run(self):
                widget = self.widget
                edges = widget.edges
                nnodes = widget.number_of_nodes
                init_temp = 0.05 if restart else 0.2
                k = widget.layout_density / 10 / np.sqrt(nnodes)
                sample_ratio =  None if nnodes < 1000 else 1000 / nnodes
                fruchterman_reingold(
                    widget.positions, edges, widget.observe_weights,
                    FR_ALLOWED_TIME, k, init_temp, sample_ratio,
                    callback_step=4, callback=self.send_update)
                self.done.emit(widget.positions)
                self.stopped.emit()

        def update(positions, progress):
            self.progressbar.advance(progress)
            self.positions = positions
            self.graph.update_coordinates()

        def done(positions):
            self.positions = positions
            self.set_buttons(running=False)
            self.graph.set_simplifications(
                self.graph.Simplifications.NoSimplifications)
            self.graph.update_coordinates()
            self.progressbar.finish()

        def thread_finished():
            self._optimizer = None
            self._animation_thread = None

        self._optimizer = LayoutOptimizer(self)
        self._animation_thread = QThread()
        self._optimizer.update.connect(update)
        self._optimizer.done.connect(done)
        self._optimizer.stopped.connect(self._animation_thread.quit)
        self._optimizer.moveToThread(self._animation_thread)
        self._animation_thread.started.connect(self._optimizer.run)
        self._animation_thread.finished.connect(thread_finished)
        self._animation_thread.start()

    def stop_optimization_and_wait(self):
        if self._animation_thread is not None:
            self._stop_optimization = True
            self._animation_thread.quit()
            self._animation_thread.wait()
            self._animation_thread = None

    def onDeleteWidget(self):
        self.stop_optimization_and_wait()
        super().onDeleteWidget()

    def send_report(self):
        if self.network is None:
            return

        self.report_items('Graph info', [
            ("Number of vertices", self.network.number_of_nodes()),
            ("Number of edges", self.network.number_of_edges()),
            ("Vertices per edge", round(self.nodes_per_edge, 3)),
            ("Edges per vertex", round(self.edges_per_node, 3)),
        ])
        if self.data is not None:
            self.report_data("Data", self.data)
        if any((self.attr_color, self.attr_shape,
                self.attr_size, self.attr_label)):
            self.report_items(
                "Visual settings",
                [("Color", self._get_caption_var_name(self.attr_color)),
                 ("Label", self._get_caption_var_name(self.attr_label)),
                 ("Shape", self._get_caption_var_name(self.attr_shape)),
                 ("Size", self._get_caption_var_name(self.attr_size))])
        self.report_plot()


def main():
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.network.network.readwrite \
        import read_pajek, transform_data_to_orange_table
    from os.path import join, dirname

    network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))
    #network = read_pajek(join(dirname(dirname(__file__)), 'networks', 'davis.net'))
    #transform_data_to_orange_table(network)
    WidgetPreview(OWNxExplorer).run(set_graph=network)

if __name__ == "__main__":
    main()
