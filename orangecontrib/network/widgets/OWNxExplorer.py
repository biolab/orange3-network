import os
from math import sqrt
from functools import wraps
from threading import Lock, Thread

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import QTimer, QSize, Qt
from AnyQt.QtWidgets import QFileDialog

import Orange
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting, SettingProvider
from Orange.widgets.widget import Input, Output

from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget

import orangecontrib.network as network
from orangecontrib.network._fr_layout import fruchterman_reingold
from orangecontrib.network.widgets.graphview import GraphView

FR_ITERATIONS = 250

def non_reentrant(func):
    """Prevent the function from reentry."""
    lock = Lock()
    @wraps(func)
    def locker(*args, **kwargs):
        if lock.acquire(False):
            try:
                return func(*args, **kwargs)
            finally:
                lock.release()
    return locker


class OWNxExplorer(OWDataProjectionWidget):
    name = "Network Explorer"
    description = "Visually explore the network and its properties."
    icon = "icons/NetworkExplorer.svg"
    priority = 6420

    class Inputs(OWDataProjectionWidget.Inputs):
        network = Input("Network", network.Graph, default=True)
        node_distances = Input("Node Distances", Orange.misc.DistMatrix)
        #node_data = OWDataProjectionWidget.Inputs.data
        #node_subset = OWDataProjectionWidget.Inputs.data_subset
        #data = data_subset = None

    class Outputs(OWDataProjectionWidget.Outputs):
        subgraph = Output("Selected sub-network", network.Graph)
        unselected_subgraph = Output("Remaining sub-network", network.Graph)
        distances = Output("Distance matrix", Orange.misc.DistMatrix)

    UserAdviceMessages = [
        widget.Message('Double clicks select connected components',
                       widget.Message.Information),
    ]

    GRAPH_CLASS = GraphView
    graph = SettingProvider(GraphView)

    randomizePositions = Setting(True)
    mark_text = Setting("")
    mark_hops = Setting(1)
    mark_min_conn = Setting(1)
    mark_max_conn = Setting(5)
    mark_most_conn = Setting(1)

    class Warning(widget.OWWidget.Warning):
        distance_matrix_mismatch = widget.Msg(
            "Distance matrix size doesn't match the number of network nodes "
            " and will be ignored.")
        no_graph_found = widget.Msg('Node data is given, graph data is missing')

    class Error(widget.OWWidget.Error):
        data_size_mismatch = widget.Msg(
            'Length of the data does not match the number of nodes.')
        network_too_large = widget.Msg('Network is too large to visualize.')
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

        super().__init__()

        self.network = None
        self.distance_matrix = None
        self.edges = None
        self.positions = None

        self.marked_nodes = None
        self.searchStringTimer = QTimer(self)
        self.searchStringTimer.timeout.connect(self.update_marks)
        self.set_mark_mode()
        self.setMinimumWidth(600)

    def sizeHint(self):
        return QSize(800, 600)

    def _add_controls(self):
        self._info_box()
        self.graph.gui.point_properties_box(self.controlArea)
        self._effects_box()
        self.graph.gui.plot_properties_box(self.controlArea)
        gui.rubber(self.controlArea)
        self.graph.box_zoom_select(self.controlArea)
        gui.auto_commit(
            self.controlArea, self, "auto_commit",
            "Send Selection", "Send Automatically")
        self._mark_box()

    def _info_box(self):
        info = gui.vBox(self.controlArea, True)
        gui.label(
            info, self,
            "Nodes: %(number_of_nodes)i (%(nodes_per_edge).2f per edge); "
            "%(nSelected)i selected")
        gui.label(
            info, self,
            "Edges: %(number_of_edges)i (%(edges_per_node).2f per node)")
        lbox = gui.hBox(info)
        self.relayout_button = gui.button(
            lbox, self, 'Re-layout', callback=self.relayout, autoDefault=False)
        self.randomize_cb = gui.checkBox(
            lbox, self, "randomizePositions", "Randomize positions")

    def _effects_box(self):
        effects_layout = self.graph.gui.effects_box(self.controlArea).layout()
        effects_layout.itemAtPosition(3, 0).widget().hide()  # hide jitter
        effects_layout.itemAtPosition(3, 1).widget().hide()
        effects_layout.addWidget(gui.widgetLabel(None, "Edge width:"), 3, 0)
        effects_layout.addWidget(
            gui.hSlider(
                None, self, 'graph.edge_width', minValue=1, maxValue=10, step=1,
                callback=self.graph.update_edges
            ), 3, 1)
        box2 = gui.hBox(None, False)
        effects_layout.addWidget(box2, 4, 1)
        self.checkbox_relative_edges = gui.checkBox(
            box2, self, 'graph.relative_edge_widths', 'Relative widths',
            callback=self.graph.update_edges)
        gui.rubber(box2)
        self.checkbox_show_weights = gui.checkBox(
            box2, self, 'graph.show_edge_weights', 'Show weights',
            callback=self.graph.set_edge_labels)

    def _mark_box(self):
        hbox = gui.hBox(None, box=True)
        self.mainArea.layout().insertWidget(0, hbox)
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
                orientation=Qt.Horizontal,
                callback=set_search_string_timer, callbackOnType=True).box

        def mark_label_starts():
            txt = self.mark_text.lower()
            if not txt:
                return None
            labels = self.get_label_data()
            if labels is None:
                return None
            return [i for i, label in enumerate(labels)
                    if label.lower().startswith(txt)]

        def mark_label_contains():
            txt = self.mark_text.lower()
            if not txt:
                return None
            labels = self.get_label_data()
            if labels is None:
                return None
            return [i for i, label in enumerate(labels) if txt in label.lower()]

        def mark_text():
            txt = self.mark_text.lower()
            if not txt or self.data is None:
                return None
            return [i for i, inst in enumerate(self.data)
                    if txt in "\x00".join(map(str, inst.list)).lower()]

        def mark_reachable():
            selected = self.graph.get_selection()
            if selected is None:
                return None
            return self.get_reachable(selected)

        def mark_close():
            selected = self.graph.get_selection()
            if selected is None:
                return
            neighbours = set(selected)
            last_round = list(neighbours)
            for _ in range(self.mark_hops):
                next_round = set()
                for neigh in last_round:
                    next_round |= set(self.network[neigh])
                neighbours |= next_round
                last_round = next_round
            neighbours -= set(selected)
            return list(neighbours)

        def mark_from_input():
            if self.subset_data is None or self.data is None:
                return None
            ids = set(self.subset_data.ids)
            return [i for i, ex in enumerate(self.data) if ex.id in ids]

        def mark_most_connections():
            n = self.mark_most_conn
            if n >= self.number_of_nodes:
                return np.arange(self.number_of_nodes)
            degrees = np.array(self.network.degree())
            min_degree = np.partition(degrees[:, 1].flatten(), -n)[-n]
            return degrees[degrees[:, 1] >= min_degree, 0]

        self.mark_criteria = [
            ("(Select criteria for marking)", None, lambda: []),
            ("Mark nodes whose label starts with", text_line(), mark_label_starts),
            ("Mark nodes whose label contains", text_line(), mark_label_contains),
            ("Mark nodes whose data that contains", text_line(), mark_text),
            ("Mark nodes reachable from selected", None, mark_reachable),

            ("Mark nodes in vicinity of selection",
             spin("mark_hops", "Number of hops:", 1, 20),
             mark_close),

            ("Mark node from subset signal", None, mark_from_input),

            ("Mark nodes with few connections",
             spin("mark_max_conn", "Max. connections:", 0, 1000),
             lambda: [node for node, degree in self.network.degree()
                      if degree <= self.mark_max_conn]),

            ("Mark nodes with many connections",
             spin("mark_min_conn", "Min. connections:", 1, 1000),
             lambda: [node for node, degree in self.network.degree()
                      if degree >= self.mark_min_conn]),

            ("Mark nodes with most connections",
             spin("mark_most_conn", "Number of marked:", 1, 1000),
             mark_most_connections),

            ("Mark nodes with more connections than any neighbour", None,
             lambda: [node for node, degree in self.network.degree()
                      if degree > max((deg for _, deg in
                                       self.network.degree(
                                           self.network[node])),
                                      default=0)]),

            ("Mark nodes with more connections than average neighbour", None,
             lambda: [node for node, degree in self.network.degree()
                      if degree > np.mean([deg for _, deg in
                                           self.network.degree(
                                               self.network[node])] or [
                                              0])])
        ]
        cb = gui.comboBox(
            None, self, "mark_mode", label=".", orientation=Qt.Horizontal,
            items=[item for item, *_ in self.mark_criteria],
            maximumContentsLength=-1, callback=self.set_mark_mode)
        vbox.layout().insertWidget(0, cb.box)

        gui.rubber(hbox)
        self.btselect = gui.button(
            hbox, self, "Select", callback=self.select_marked)
        self.btadd = gui.button(
            hbox, self, "Add to Selection", callback=self.select_add_marked)
        self.btgroup = gui.button(
            hbox, self, "Add New Group", callback=self.select_as_group)

    @non_reentrant
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

    @non_reentrant
    def update_marks(self):
        if self.network is None:
            return
        to_mark = self.mark_criteria[self.mark_mode][2]()
        if to_mark is None or not len(to_mark):
            self.marked_nodes = None
        else:
            self.marked_nodes = np.asarray(to_mark)
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
        self.update_selection_buttons()
        self.update_marks()

    def select_marked(self):
        self.graph.selection_select(self.marked_nodes)
        self.set_mark_mode(0)

    def select_add_marked(self):
        self.graph.selection_append(self.marked_nodes)
        self.set_mark_mode(0)

    def select_as_group(self):
        self.graph.selection_new_group(self.marked_nodes)
        self.set_mark_mode(0)

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

        def compute_labels():
            self.number_of_nodes = graph.number_of_nodes()
            self.number_of_edges = graph.number_of_edges()
            self.edges_per_node = self.number_of_edges / self.number_of_nodes
            self.nodes_per_edge = \
                self.number_of_nodes / max(1, self.number_of_edges)

        if not graph:
            return set_graph_none()
        if graph.number_of_nodes() < 2:
            return set_graph_none(self.Error.single_node_graph)
        if graph.number_of_nodes() + graph.number_of_edges() > 30000:
            return set_graph_none(self.Error.network_too_large)
        self.Error.clear()

        self.network = graph
        compute_labels()
        self.positions = None

    def handleNewSignals(self):
        network = self.network

        def set_actual_data():
            self.Error.data_size_mismatch.clear()
            self.Warning.no_graph_found.clear()
            self._invalid_data = False
            if network is None:
                if self.data is not None:
                    self.Warning.no_graph_found()
                return
            if self.data is None:
                # Replicate the necessary parts of set_data
                self.data = network.items()
                self.init_attr_values()
                self.openContext(self.data)
                self.cb_class_density.setEnabled(self.can_draw_density())
            else:
                if len(self.data) != self.number_of_nodes:
                    self.Error.data_size_mismatch()
                    self._invalid_data = True

        def set_actual_edges():
            def set_checkboxes(value):
                self.checkbox_show_weights.setEnabled(value)
                self.checkbox_relative_edges.setEnabled(value)

            self.Warning.distance_matrix_mismatch.clear()

            if self.network is None:
                self.edges = None
                return set_checkboxes(False)

            set_checkboxes(True)
            row, col, data = zip(*network.edges(data='weight'))
            self.edges = sp.coo_matrix((data, (row, col)))
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

        set_actual_data()
        if self.positions is None:
            set_actual_edges()
            self.set_random_positions()
            self.graph.reset_graph()
            self.relayout()
        else:
            self.graph.update_point_props()
        self.update_selection_buttons()

    def set_random_positions(self):
        self.positions = np.random.uniform(size=(self.number_of_nodes, 2))

    def get_reachable(self, initial):
        to_check = list(initial)
        reachable = set(to_check)
        for node in to_check:
            new_checks = set(self.network[node]) - reachable
            to_check += new_checks
            reachable |= new_checks
        return list(reachable)

    def save_network(self):
        # TODO: this was never reviewed since Orange2
        if self.view is None or self.network is None:
            return

        filename = QFileDialog.getSaveFileName(
            self, 'Save Network', '',
            'NetworkX graph as Python pickle (*.gpickle)\n'
            'NetworkX edge list (*.edgelist)\n'
            'Pajek network (*.net *.pajek)\n'
            'GML network (*.gml)')
        if filename:
            _, ext = os.path.splitext(filename)
            if not ext: filename += ".net"
            items = self.network.items()
            for i in range(self.network.number_of_nodes()):
                graph_node = self.network.node[i]
                plot_node = self.networkCanvas.networkCurve.nodes()[i]

                if items is not None:
                    ex = items[i]
                    if 'x' in ex.domain: ex['x'] = plot_node.x()
                    if 'y' in ex.domain: ex['y'] = plot_node.y()

                graph_node['x'] = plot_node.x()
                graph_node['y'] = plot_node.y()

            network.readwrite.write(self.network, filename)

    def send_data(self):
        return
        super().send_data()

        Outputs = self.Outputs
        if self.network is None:
            Outputs.subgraph.send(None)
            Outputs.unselected_subgraph.send(None)
            Outputs.distances.send(None)
            return

        selection = self.graph.get_selection()
        if selection is None:
            Outputs.subgraph.send(None)
            Outputs.unselected_subgraph.send(self.network)
            Outputs.distances.send(None)
            return

        sel_indices = np.nonzero(selection)
        unsel_indices = np.nonzero(selection == 0)
        Outputs.subgraph.send(self.network.subgraph(sel_indices))
        Outputs.unselected_subgraph.send(self.network.subgraph(unsel_indices))
        distances = self.distance_matrix
        if distances is None:
            Outputs.distances.send(None)
        else:
            Outputs.distances.send(distances.submatrix(sorted(sel_indices)))

    def get_coordinates_data(self):
        if self.positions is not None:
            return self.positions.T
        else:
            return None, None

    def get_edges(self):
        return self.edges

    def get_marked_nodes(self):
        return self.marked_nodes

    # TODO: Stop relayout if new data is received
    def relayout(self):
        if self.edges is None:
            return
        if self.randomizePositions:
            self.set_random_positions()
        self.progressbar = gui.ProgressBar(self, FR_ITERATIONS)
        self.relayout_button.setDisabled(True)

        Simplifications = self.graph.Simplifications
        self.graph.set_simplifications(
            Simplifications.NoDensity
            + Simplifications.NoLabels * (self.graph.labels is not None
                                          and len(self.graph.labels) > 20)
            + Simplifications.NoEdges * (self.number_of_edges > 1000))

        if self.number_of_nodes + self.number_of_edges > 20000:
            iterations = 5
            callback = None
        else:
            iterations = FR_ITERATIONS

            def callback(positions, progress):
                self.progressbar.advance(progress)
                self.positions = np.array(positions)
                self.graph.update_coordinates()
                return True

        def done():
            self.graph.set_simplifications(Simplifications.NoSimplifications)
            self.graph.update_coordinates()
            self.relayout_button.setDisabled(False)
            self.progressbar.finish()

        class AnimationThread(Thread):
            def __init__(self):
                super().__init__()
                self.daemon = True

            def run(_):
                edges = self.edges
                self.positions = np.array(fruchterman_reingold(
                    edges.data, edges.row, edges.col,
                    1 / sqrt(self.number_of_nodes),  # k
                    self.positions,
                    np.array([], dtype=np.int32),  # fixed
                    iterations,
                    0.1,  # sample ratio
                    callback, 0.5))
                done()

        AnimationThread().start()

    def send_report(self):
        self.report_data("Data", self.network.items())
        self.report_items('Graph info', [
            ("Number of vertices", self.network.number_of_nodes()),
            ("Number of edges", self.network.number_of_edges()),
            ("Vertices per edge", "%.3f" % self.nodes_per_edge),
            ("Edges per vertex", "%.3f" % self.edges_per_node),
        ])
        if self.node_color_attr or self.node_size_attr or self.node_label_attrs:
            self.report_items("Visual settings", [
                ("Vertex color", self.colorCombo.currentText()),
                ("Vertex size", str(self.nodeSizeCombo.currentText())),
            ])
        self.report_plot("Graph", self.view)


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    a = QApplication([])
    ow = OWNxExplorer()
    ow.show()

    def set_network(data, id=None):
        ow.set_graph(data)

    import OWNxFile
    from os.path import join, dirname
    owFile = OWNxFile.OWNxFile()
    owFile.Outputs.network.send = set_network
    # owFile.openNetFile(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))
    owFile.openNetFile(join(dirname(dirname(__file__)), 'networks', 'leu_by_pmid.net'))
    ow.handleNewSignals()
    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()
