import os
from functools import wraps
from threading import Lock
from xml.sax.saxutils import escape

import numpy as np

from AnyQt.QtCore import QTimer, QSize, Qt, QItemSelection, QItemSelectionRange
from AnyQt.QtGui import QBrush, QColor
from AnyQt.QtWidgets import QListWidget, QFileDialog

import Orange
from Orange.util import scale
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator, GradientPaletteGenerator
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.itemmodels import VariableListModel

from Orange.data import Table, DiscreteVariable, StringVariable
from Orange.widgets.visualize.owdistributions import ScatterPlotItem
from Orange.widgets.visualize.owscatterplotgraph import PaletteItemSample, DiscretizedScale

import orangecontrib.network as network
from orangecontrib.network.widgets.graphview import GraphView, Node, FR_ITERATIONS, LegendItem


def non_reentrant(func):
    """Prevent the function from reentry."""
    lock = Lock()
    @wraps(func)
    def locker(*args, **kwargs):
        if lock.acquire(False):
            try: return func(*args, **kwargs)
            finally: lock.release()
    return locker


CONTINUOUS_PALETTE = GradientPaletteGenerator('#00ffff', '#550066')
MIN_NODE_SIZE = 1
MAX_NODE_SIZE = 10


class SelectionMode:
    NONE,       \
    SEARCH,     \
    NEIGHBORS,  \
    AT_LEAST_N, \
    AT_MOST_N,  \
    ANY_NEIGH,  \
    AVG_NEIGH,  \
    MOST_CONN,  \
    FROM_INPUT  \
    = range(9)  # FML


class OWNxExplorer(widget.OWWidget):
    name = "Network Explorer"
    description = "Visually explore the network and its properties."
    icon = "icons/NetworkExplorer.svg"
    priority = 6420

    class Inputs:
        network = Input("Network", network.Graph, default=True)
        node_subset = Input("Node Subset", Table)
        node_data = Input("Node Data", Table)
        node_distances = Input("Node Distances", Orange.misc.DistMatrix)

    class Outputs:
        subgraph = Output("Selected sub-network", network.Graph)
        unselected_subgraph = Output("Remaining sub-network", network.Graph)
        distances = Output("Distance matrix", Orange.misc.DistMatrix)
        selected = Output("Selected items", Table)
        highlighted = Output("Highlighted items", Table)
        remaining = Output("Remaining items", Table)

    UserAdviceMessages = [
        widget.Message('When selecting nodes on the Marking tab, '
                       'press <b><tt>Enter</tt></b> key to add '
                       '<b><font color="{}">highlighted</font></b> nodes to '
                       '<b><font color="{}">selection</font></b>.'
                       .format(Node.Pen.HIGHLIGHTED.color().name(),
                               Node.Pen.SELECTED.color().name()),
                       'marking-info',
                       widget.Message.Information),
        widget.Message('Left-click to select nodes '
                       '(hold <b><tt>Shift</tt></b> to append to selection). '
                       'Right-click to pan/move the view. Scroll to zoom.',
                       'mouse-info',
                       widget.Message.Information),
    ]

    settingsHandler = DomainContextHandler()

    do_auto_commit = Setting(True)
    selectionMode = Setting(SelectionMode.FROM_INPUT)
    tabIndex = Setting(0)
    showEdgeWeights = Setting(False)
    relativeEdgeWidths = Setting(False)
    randomizePositions = Setting(True)
    invertNodeSize = Setting(False)
    markDistance = Setting(1)
    markSearchString = Setting("")
    markNBest = Setting(1)
    markNConnections = Setting(2)

    point_width = Setting(10)
    edge_width = Setting(1)
    attr_size = ContextSetting(None)
    attr_color = ContextSetting(None)
    attrs_label = ContextSetting({})
    attrs_tooltip = ContextSetting({})
    graph_name = 'view'

    class Warning(widget.OWWidget.Warning):
        distance_matrix_size = widget.Msg("Distance matrix size doesn't match the number of network nodes. Not using it.")
        no_graph_found = widget.Msg('No graph found!')
        no_graph_or_items = widget.Msg('No graph provided or no items attached to the graph.')

    class Error(widget.OWWidget.Error):
        instance_for_each_node = widget.Msg('Items table must have one instance for each network node.')
        network_too_large = widget.Msg('Network is too large to visualize. Sorry.')

    def __init__(self):
        super().__init__()
        #self.contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", selected="node_label_attrs"), ContextField("attributes", selected="tooltipAttributes"), "color"])}

        self.view = GraphView(self)
        self.mainArea.layout().addWidget(self.view)

        self.graph_attrs = []

        self.acceptingEnterKeypress = False

        self.node_label_attrs = []
        self.tooltipAttributes = []
        self.searchStringTimer = QTimer(self)
        self.markInputItems = None
        self.node_color_attr = 0
        self.node_size_attr = 0

        self.nHighlighted = 0
        self.nSelected = 0
        self.verticesPerEdge = 0
        self.edgesPerVertex = 0

        self.items_matrix = None
        self.number_of_nodes_label = 0
        self.number_of_edges_label = 0

        self.graph = None

        self.setMinimumWidth(600)

        self.tabs = gui.tabWidget(self.controlArea)
        self.displayTab = gui.createTabPage(self.tabs, "Display")
        self.markTab = gui.createTabPage(self.tabs, "Marking")

        def on_tab_changed(index):
            self.tabIndex = index
            self.set_selection_mode()
        self.tabs.currentChanged.connect(on_tab_changed)
        self.tabs.setCurrentIndex(self.tabIndex)

        ib = gui.widgetBox(self.displayTab, "Info")
        gui.label(ib, self, "Nodes: %(number_of_nodes_label)i (%(verticesPerEdge).2f per edge)")
        gui.label(ib, self, "Edges: %(number_of_edges_label)i (%(edgesPerVertex).2f per node)")

        box = gui.widgetBox(self.displayTab, "Nodes")

        self.relayout_button = gui.button(box, self, 'Re-layout',
                                          callback=self.relayout, autoDefault=False)
        self.randomize_cb = gui.checkBox(box, self, "randomizePositions", "Randomize positions")
        self.view.positionsChanged.connect(lambda positions, progress:
                                           self.progressbar.widget.progressBarSet(int(round(100 * progress))))
        def animationFinished():
            self.relayout_button.setEnabled(True)
            self.progressbar.finish()
        self.view.animationFinished.connect(animationFinished)

        self.color_model = VariableListModel(placeholder="(Same color)")
        self.color_combo = gui.comboBox(
            box, self, "attr_color", label='Color:',
            orientation='horizontal', callback=self.set_node_colors,
            model=self.color_model)

        self.size_model = VariableListModel(placeholder="(Same size)")
        self.size_combo = gui.comboBox(
            box, self, "attr_size",
            label='Size:', orientation='horizontal',
            callback=self.set_node_sizes, model=self.size_model)
        gui.hSlider(box, self, 'point_width', label="Symbol size:   ", minValue=1,
                    maxValue=10, step=1, createLabel=False,
                    callback=self.set_node_sizes)
        hb = gui.widgetBox(box, orientation="horizontal")
        hb.layout().addStretch(1)
        self.invertNodeSizeCheck = gui.checkBox(
            hb, self, "invertNodeSize", "Invert",
            callback=self.set_node_sizes)

        hb = gui.widgetBox(self.displayTab, box="Node labels | tooltips",
                           orientation="horizontal", addSpace=False)
        self.attListBox = gui.listBox(
            hb, self, "node_label_attrs", "graph_attrs",
            selectionMode=QListWidget.MultiSelection,
            sizeHint=QSize(100, 100),
            callback=self._on_node_label_attrs_changed)
        self.tooltipListBox = gui.listBox(
            hb, self, "tooltipAttributes", "graph_attrs",
            selectionMode=QListWidget.MultiSelection,
            sizeHint=QSize(100, 100),
            callback=self._clicked_tooltip_lstbox)

        eb = gui.widgetBox(self.displayTab, "Edges", orientation="vertical")
        self.checkbox_relative_edges = gui.checkBox(
            eb, self, 'relativeEdgeWidths', 'Relative edge widths',
            callback=self.set_edge_sizes)
        gui.hSlider(eb, self, 'edge_width', label="Edge width: ", minValue=1,
                    maxValue=10, step=1, createLabel=False,
                    callback=self.set_edge_sizes)
        self.checkbox_show_weights = gui.checkBox(
            eb, self, 'showEdgeWeights', 'Show edge weights',
            callback=self.set_edge_labels)

        ib = gui.widgetBox(self.markTab, "Info", orientation="vertical")
        gui.label(ib, self, "Nodes: %(number_of_nodes_label)i")
        gui.label(ib, self, "Selected: %(nSelected)i")
        gui.label(ib, self, "Highlighted: %(nHighlighted)i")
        def on_selection_change():
            self.nSelected = len(self.view.getSelected())
            self.nHighlighted = len(self.view.getHighlighted())
            self.set_selection_mode()
            self.commit()
        self.view.selectionChanged.connect(on_selection_change)

        ib = gui.widgetBox(self.markTab, "Highlight nodes ...")
        ribg = gui.radioButtonsInBox(ib, self, "selectionMode", callback=self.set_selection_mode)
        gui.appendRadioButton(ribg, "None")
        gui.appendRadioButton(ribg, "... whose attributes contain:")
        self.ctrlMarkSearchString = gui.lineEdit(gui.indentedBox(ribg), self, "markSearchString", callback=self._set_search_string_timer, callbackOnType=True)
        self.searchStringTimer.timeout.connect(self.set_selection_mode)

        gui.appendRadioButton(ribg, "... neighbours of selected, ≤ N hops away")
        ib = gui.indentedBox(ribg, orientation=0)
        self.ctrlMarkDistance = gui.spin(ib, self, "markDistance", 1, 100, 1, label="Hops:",
            callback=lambda: self.set_selection_mode(SelectionMode.NEIGHBORS))
        ib.layout().addStretch(1)
        gui.appendRadioButton(ribg, "... with at least N connections")
        gui.appendRadioButton(ribg, "... with at most N connections")
        ib = gui.indentedBox(ribg, orientation=0)
        self.ctrlMarkNConnections = gui.spin(ib, self, "markNConnections", 0, 1000000, 1, label="Connections:",
            callback=lambda: self.set_selection_mode(SelectionMode.AT_MOST_N if self.selectionMode == SelectionMode.AT_MOST_N else SelectionMode.AT_LEAST_N))
        ib.layout().addStretch(1)
        gui.appendRadioButton(ribg, "... with more connections than any neighbor")
        gui.appendRadioButton(ribg, "... with more connections than average neighbor")
        gui.appendRadioButton(ribg, "... with most connections")
        ib = gui.indentedBox(ribg, orientation=0)
        self.ctrlMarkNumber = gui.spin(ib, self, "markNBest", 1, 1000000, 1,
                                       label="Number of nodes:",
                                       callback=lambda: self.set_selection_mode(SelectionMode.MOST_CONN))
        ib.layout().addStretch(1)
        self.markInputRadioButton = gui.appendRadioButton(ribg, "... from Node Subset input signal")
        self.markInputRadioButton.setEnabled(True)

        gui.auto_commit(ribg, self, 'do_auto_commit', 'Output changes')
        self.markTab.layout().addStretch(1)

        self.set_graph(None)
        self.set_selection_mode()

    def sizeHint(self):
        return QSize(800, 600)

    def commit(self):
        self.send_data()

    @Inputs.node_distances
    def set_items_distance_matrix(self, matrix):
        assert matrix is None or isinstance(matrix, Orange.misc.DistMatrix)
        self.items_matrix = matrix
        self.relayout()

    def _set_search_string_timer(self):
        self.selectionMode = SelectionMode.SEARCH
        self.searchStringTimer.stop()
        self.searchStringTimer.start(300)

    def switchTab(self, index=None):
        index = index or self.tabs.currentIndex()
        curTab = self.tabs.widget(index)
        self.acceptingEnterKeypress = False
        if curTab == self.markTab and self.selectionMode != SelectionMode.NONE:
            self.acceptingEnterKeypress = True

    @non_reentrant
    def set_selection_mode(self, selectionMode=None):
        self.searchStringTimer.stop()
        selectionMode = self.selectionMode = selectionMode or self.selectionMode
        self.switchTab()
        if (self.graph is None or
            self.tabs.widget(self.tabs.currentIndex()) != self.markTab and selectionMode != SelectionMode.FROM_INPUT):
            return

        if selectionMode == SelectionMode.NONE:
            self.view.setHighlighted([])
        elif selectionMode == SelectionMode.SEARCH:
            table, txt = self.graph.items(), self.markSearchString.lower()
            if not table or not txt: return
            toMark = set(i for i, instance in enumerate(table)
                         if txt in " ".join(map(str, instance.list)).lower())
            self.view.setHighlighted(toMark)
        elif selectionMode == SelectionMode.NEIGHBORS:
            selected = set(self.view.getSelected())
            neighbors = selected.copy()
            for _ in range(self.markDistance):
                for neigh in list(neighbors):
                    neighbors |= set(self.graph[neigh].keys())
            neighbors -= selected
            self.view.setHighlighted(neighbors)
        elif selectionMode == SelectionMode.AT_LEAST_N:
            self.view.setHighlighted(
                set(node for node, degree in self.graph.degree()
                    if degree >= self.markNConnections))
        elif selectionMode == SelectionMode.AT_MOST_N:
            self.view.setHighlighted(
                set(node for node, degree in self.graph.degree()
                    if degree <= self.markNConnections))
        elif selectionMode == SelectionMode.ANY_NEIGH:
            self.view.setHighlighted(
                set(node for node, degree in self.graph.degree()
                    if degree > max(dict(self.graph.degree(self.graph[node])).values(), default=0)))
        elif selectionMode == SelectionMode.AVG_NEIGH:
            self.view.setHighlighted(
                set(node for node, degree in self.graph.degree()
                    if degree > np.nan_to_num(np.mean(list(dict(self.graph.degree(self.graph[node])).values())))))
        elif selectionMode == SelectionMode.MOST_CONN:
            degrees = np.array(sorted(self.graph.degree(), key=lambda i: i[1], reverse=True))
            cut_ind = max(1, min(self.markNBest, self.graph.number_of_nodes()))
            cut_degree = degrees[cut_ind - 1, 1]
            toMark = set(degrees[degrees[:, 1] >= cut_degree, 0])
            self.view.setHighlighted(toMark)
        elif selectionMode == SelectionMode.FROM_INPUT:
            tomark = {}
            if self.markInputItems:
                ids = set(self.markInputItems.ids)
                tomark = {x for x in self.graph
                          if self.graph.items()[x].id in ids}
            self.view.setHighlighted(tomark)

    def keyReleaseEvent(self, ev):
        """On Enter, expand the selected set with the highlighted"""
        if (not self.acceptingEnterKeypress or
            ev.key() not in (Qt.Key_Return, Qt.Key_Enter)):
            super().keyReleaseEvent(ev)
            return
        highlighted = self.view.getHighlighted()
        self.view.setSelected(highlighted, extend=True)
        self.view.setHighlighted([])
        self.set_selection_mode()

    def save_network(self):
        # TODO: this was never reviewed since Orange2
        if self.view is None or self.graph is None:
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
            items = self.graph.items()
            for i in range(self.graph.number_of_nodes()):
                graph_node = self.graph.node[i]
                plot_node = self.networkCanvas.networkCurve.nodes()[i]

                if items is not None:
                    ex = items[i]
                    if 'x' in ex.domain: ex['x'] = plot_node.x()
                    if 'y' in ex.domain: ex['y'] = plot_node.y()

                graph_node['x'] = plot_node.x()
                graph_node['y'] = plot_node.y()

            network.readwrite.write(self.graph, filename)

    def send_data(self):
        if not self.graph:
            for output in dir(self.Outputs):
                if not output.startswith('__'):
                    getattr(self.Outputs, output).send(None)
            return
        selected = self.view.getSelected()
        self.Outputs.subgraph.send(self.graph.subgraph(selected) if selected else None)
        self.Outputs.unselected_subgraph.send(
                  self.graph.subgraph(self.view.getUnselected()) if selected else self.graph)
        self.Outputs.distances.send(
                  self.items_matrix.submatrix(sorted(selected))
                  if self.items_matrix is not None and selected else None)
        items = self.graph.items()
        if not items:
            self.Outputs.selected.send(None)
            self.Outputs.highlighted.send(None)
            self.Outputs.remaining.send(None)
        else:
            highlighted = self.view.getHighlighted()
            self.Outputs.selected.send(items[sorted(selected), :] if selected else None)
            self.Outputs.highlighted.send(items[sorted(highlighted), :] if highlighted else None)
            remaining = sorted(set(self.graph) - set(selected) - set(highlighted))
            self.Outputs.remaining.send(items[remaining, :] if remaining else None)

    def _set_combos(self):
        self._clear_combos()
        self.graph_attrs = self.graph.items_vars()

        self.color_model[:] = [None] + [v for v in self.graph_attrs if v.is_primitive()]
        self.size_model[:] = [None] + [v for v in self.graph_attrs if v.is_continuous]
        self.size_combo.setDisabled(not self.graph_attrs)
        self.color_combo.setDisabled(not self.graph_attrs)
        self.set_node_sizes()
        self.set_node_colors()
        self.set_edge_sizes()

        for columns, box in ((self.attrs_label, self.attListBox),
                             (self.attrs_tooltip, self.tooltipListBox)):
            columns = [var.name for var in columns]
            if columns:
                selection = QItemSelection()
                model = box.model()
                for i in range(box.count()):
                    if str(box.item(i).text()) in columns:
                        selection.append(QItemSelectionRange(model.index(i, 0)))
                selmodel = box.selectionModel()
                selmodel.select(selection, selmodel.Select | selmodel.Clear)
            else:
                box.selectionModel().clearSelection()
        self._on_node_label_attrs_changed()
        self._clicked_tooltip_lstbox()

    def _clear_combos(self):
        self.graph_attrs = []
        self.color_combo.clear()
        self.size_combo.clear()

    def set_graph_none(self):
        self.graph = None
        self.graph_base = None
        self._clear_combos()
        self.number_of_nodes_label = 0
        self.number_of_edges_label = 0
        self.verticesPerEdge = 0
        self.edgesPerVertex = 0
        self._items = None
        self.view.set_graph(None)

    @Inputs.network
    def set_graph(self, graph):
        if not graph:
            return self.set_graph_none()
        if graph.number_of_nodes() < 2:
            self.set_graph_none()
            self.information('I\'m not really in a mood to visualize just one node. Try again tomorrow.')
            return
        if graph.number_of_nodes() + graph.number_of_edges() > 30000:
            self.set_graph_none()
            self.Error.network_too_large()
            return
        self.information()
        self.closeContext()

        all_edges_equal = bool(1 == len(set(w for u,v,w in graph.edges(data='weight'))))
        self.checkbox_show_weights.setEnabled(not all_edges_equal)
        self.checkbox_relative_edges.setEnabled(not all_edges_equal)

        self.graph_base = graph
        self.graph = graph.copy()
        # Set items table from the separate signal
        if self._items: self.set_items(self._items)

        self.view.set_graph(self.graph, relayout=False)

        # Set labels
        self.number_of_nodes_label = self.graph.number_of_nodes()
        self.number_of_edges_label = self.graph.number_of_edges()
        self.verticesPerEdge = self.graph.number_of_nodes() / max(1, self.graph.number_of_edges())
        self.edgesPerVertex = self.graph.number_of_edges() / max(1, self.graph.number_of_nodes())

        self._set_combos()
        if self.graph.items():
            self.openContext(self.graph.items().domain)
        self.Error.clear()

        self.set_selection_mode()
        self.randomizePositions = True
        self.relayout()

    @Inputs.node_data
    def set_items(self, items=None):
        self._items = items
        if items is None:
            return self.set_graph(self.graph_base)
        if not self.graph:
            self.Warning.no_graph_found()
            return
        self.Warning.clear()
        if len(items) != self.graph.number_of_nodes():
            self.Error.instance_for_each_node()
            return
        self.Error.instance_for_each_node.clear()
        self.graph.set_items(items)
        self._set_combos()

    @Inputs.node_subset
    def set_marking_items(self, items):
        self.markInputRadioButton.setEnabled(False)
        self.markInputItems = items

        self.Warning.clear()

        if self.selectionMode == SelectionMode.FROM_INPUT and \
                (items is None or self.graph is None or self.graph.items() is None):
            self.selectionMode = SelectionMode.NONE

        if items is None:
            self.view.selectionChanged.emit()
            return

        if self.graph is None or self.graph.items() is None:
            self.Warning.no_graph_or_items()
            return

        if len(items) > 0:
            self.markInputRadioButton.setEnabled(True)
        self.view.selectionChanged.emit()

    def relayout(self):
        if self.graph is None or self.graph.number_of_nodes() <= 1:
            return
        self.progressbar = gui.ProgressBar(self, FR_ITERATIONS)

        distmatrix = self.items_matrix
        if distmatrix is not None and distmatrix.shape[0] != self.graph.number_of_nodes():
            self.Warning.distance_matrix_size()
            distmatrix = None
        self.Warning.distance_matrix_size.clear()

        self.relayout_button.setDisabled(True)
        self.view.relayout(randomize=self.randomizePositions, weight=distmatrix)

    def _on_node_label_attrs_changed(self):
        if not self.graph: return
        attributes = self.attrs_label = [self.graph_attrs[i] for i in self.node_label_attrs]
        if attributes:
            table = self.graph.items()
            if not table: return
            for i, node in enumerate(self.view.nodes):
                text = ', '.join(map(str, table[i, attributes][0].list))
                node.setText(text)
        else:
            for node in self.view.nodes:
                node.setText('')

    def _clicked_tooltip_lstbox(self):
        if not self.graph: return
        attributes = self.attrs_tooltip = [self.graph_attrs[i] for i in self.tooltipAttributes]
        if attributes:
            table = self.graph.items()
            if not table: return
            assert self.view.nodes
            for i, node in enumerate(self.view.nodes):
                node.setTooltip(lambda row=i, attributes=attributes, table=table:
                    '<br>'.join('<b>{.name}:</b> {}'.format(i[0], str(i[1]).replace('<', '&lt;'))
                                for i in zip(attributes, table[row, attributes][0].list))
                )
        else:
            for node in self.view.nodes:
                node.setTooltip(None)

    def set_edge_labels(self):
        if not self.graph:
            return
        if self.showEdgeWeights:
            weights = (str(w or '') for u, v, w in self.graph.edges(data='weight'))
        else:
            weights = ('' for i in range(self.graph.number_of_edges()))
        for edge, weight in zip(self.view.edges, weights):
            edge.setText(weight)

    def set_node_colors(self):
        if not self.graph: return
        attribute = self.attr_color
        assert not attribute or isinstance(attribute, Orange.data.Variable)
        if self.view.legend is not None:
            self.view.scene().removeItem(self.view.legend)
            self.view.legend.clear()
        else:
            self.view.legend = LegendItem()
            self.view.legend.set_parent(self.view)
        if not attribute:
            for node in self.view.nodes:
                node.setColor(None)
            return
        table = self.graph.items()
        if not table: return
        if attribute in table.domain.class_vars:
            values = table[:, attribute].Y
            if values.ndim > 1:
                values = values.T
        elif attribute in table.domain.metas:
            values = table[:, attribute].metas[:, 0]
        elif attribute in table.domain.attributes:
            values = table[:, attribute].X[:, 0]
        else: raise RuntimeError("Shouldn't be able to select this column")
        if attribute.is_continuous:
            colors = CONTINUOUS_PALETTE[scale(values)]
            label = PaletteItemSample(CONTINUOUS_PALETTE,
                                      DiscretizedScale(np.nanmin(values), np.nanmax(values)))
            self.view.legend.addItem(label, "")
            self.view.legend.setGeometry(label.boundingRect())
        elif attribute.is_discrete:
            DISCRETE_PALETTE = ColorPaletteGenerator(len(attribute.values))
            colors = DISCRETE_PALETTE[values]
            for value, color in zip(attribute.values, DISCRETE_PALETTE):
                self.view.legend.addItem(
                    ScatterPlotItem(pen=Node.Pen.DEFAULT, brush=QBrush(QColor(color)), size=10,
                                    symbol="o"), escape(value))
        for node, color in zip(self.view.nodes, colors):
            node.setColor(color)
        self.view.scene().addItem(self.view.legend)
        self.view.legend.geometry_changed()

    def set_node_sizes(self):
        self.invertNodeSizeCheck.setDisabled(not self.attr_size)

        if not self.graph:
            return
        table = self.graph.items()
        if table is None:
            return

        try:
            a = table.get_column_view(self.attr_size)[0]
            values = a.copy()
        except Exception:
            for node in self.view.nodes:
                node.setSize(MIN_NODE_SIZE * self.point_width)
            return

        if self.invertNodeSize:
            values += np.nanmin(values) + 1
            values = 1/values
        nodemin, nodemax = np.nanmin(values), np.nanmax(values)
        if nodemin == nodemax:
            # np.polyfit borks on this condition
            sizes = (MIN_NODE_SIZE for _ in range(len(self.view.nodes)))
        else:
            k, n = np.polyfit([nodemin, nodemax],
                              [MIN_NODE_SIZE, MAX_NODE_SIZE], 1)
            sizes = values * k + n
            sizes[np.isnan(sizes)] = np.nanmean(sizes)
        for node, size in zip(self.view.nodes, sizes):
            node.setSize(size * self.point_width)

    def set_edge_sizes(self):
        if not self.graph: return
        if self.relativeEdgeWidths:
            widths = [self.graph.adj[u][v].get('weight', 1)
                      for u, v in self.graph.edges()]
            widths = scale(widths, .7, 8) * np.log2(self.edge_width/4 + 1)
        else:
            widths = (.7 * self.edge_width for _ in range(self.graph.number_of_edges()))
        for edge, width in zip(self.view.edges, widths):
            edge.setSize(width)

    def send_report(self):
        self.report_data("Data", self.graph.items())
        self.report_items('Graph info', [
            ("Number of vertices", self.graph.number_of_nodes()),
            ("Number of edges", self.graph.number_of_edges()),
            ("Vertices per edge", "%.3f" % self.verticesPerEdge),
            ("Edges per vertex", "%.3f" % self.edgesPerVertex),
        ])
        if self.node_color_attr or self.node_size_attr or self.node_label_attrs:
            self.report_items("Visual settings", [
                ("Vertex color", self.colorCombo.currentText()),
                ("Vertex size", str(self.nodeSizeCombo.currentText()) + " (inverted)" if self.invertNodeSize else ""),
                ("Labels", ", ".join(self.graph_attrs[i].name for i in self.node_label_attrs)),
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
    owFile.openNetFile(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))

    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()
