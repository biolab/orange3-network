import os
from operator import itemgetter, add
from functools import reduce, wraps
from itertools import chain
from threading import Lock

import numpy as np

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import Orange
from Orange.util import scale
from Orange.widgets import gui, widget, settings
from Orange.widgets.utils.colorpalette import ColorPaletteDlg
from Orange.widgets.unsupervised.owmds import torgerson as MDS
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator, GradientPaletteGenerator

from Orange.data import Table, Domain, DiscreteVariable, StringVariable
import orangecontrib.network as network
from orangecontrib.network.widgets.graphview import GraphView, Node, FR_ITERATIONS


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


class Output:
    SUBGRAPH = 'Selected sub-network'
    DISTANCE = 'Distance matrix'
    SELECTED = 'Selected items'
    HIGHLIGHTED = 'Highlighted items'
    REMAINING = 'Remaining items'
    all = (SUBGRAPH, DISTANCE, SELECTED, HIGHLIGHTED, REMAINING)


class OWNxExplorer(widget.OWWidget):
    name = "Network Explorer"
    description = "Visually explore the network and its properties."
    icon = "icons/NetworkExplorer.svg"
    priority = 6420

    inputs = [
        ("Network", network.Graph, "set_graph", widget.Default),
        ("Node Subset", Table, 'set_marking_items'),
        ("Node Data", Table, "set_items"),
        ("Node Distances", Orange.misc.DistMatrix, "set_items_distance_matrix"),
    ]

    outputs = [(Output.SUBGRAPH, network.Graph),
               (Output.DISTANCE, Orange.misc.DistMatrix),
               (Output.SELECTED, Table),
               (Output.HIGHLIGHTED, Table),
               (Output.REMAINING, Table)]

    settingsList = ["lastVertexSizeColumn", "lastColorColumn",
                    "lastLabelColumns", "lastTooltipColumns",]
    # TODO: set settings

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

    do_auto_commit = settings.Setting(True)
    maxNodeSize = settings.Setting(50)
    minNodeSize = settings.Setting(8)
    selectionMode = settings.Setting(0)
    tabIndex = settings.Setting(0)
    showEdgeWeights = settings.Setting(False)
    relativeEdgeWidths = settings.Setting(False)
    invertNodeSize = settings.Setting(False)
    markDistance = settings.Setting(1)
    markSearchString = settings.Setting("")
    markNBest = settings.Setting(1)
    markNConnections = settings.Setting(2)

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

        self.lastVertexSizeColumn = ''
        self.lastColorColumn = ''
        self.lastLabelColumns = set()
        self.lastTooltipColumns = set()

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
        self.view.positionsChanged.connect(lambda _: self.progressbar.advance())
        def animationFinished():
            self.relayout_button.setEnabled(True)
            self.progressbar.finish()
        self.view.animationFinished.connect(animationFinished)

        self.colorCombo = gui.comboBox(
            box, self, "node_color_attr", label='Color:',
            orientation='horizontal', callback=self.set_node_colors)

        self.invertNodeSizeCheck = self.maxNodeSizeSpin = QWidget()  # Forward declaration
        self.nodeSizeCombo = gui.comboBox(
            box, self, "node_size_attr",
            label='Size:',
            orientation='horizontal',
            callback=self.set_node_sizes)
        hb = gui.widgetBox(box, orientation="horizontal")
        hb.layout().addStretch(1)
        self.minNodeSizeSpin = gui.spin(
            hb, self, "minNodeSize", 1, 50, step=1, label="Min:",
            callback=self.set_node_sizes)
        self.minNodeSizeSpin.setValue(8)
        gui.separator(hb)
        self.maxNodeSizeSpin = gui.spin(
            hb, self, "maxNodeSize", 10, 200, step=5, label="Max:",
            callback=self.set_node_sizes)
        self.maxNodeSizeSpin.setValue(50)
        gui.separator(hb)
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
        self.markInputRadioButton = gui.appendRadioButton(ribg, "... given in the ItemSubset input signal")
        self.markInput = 0
        ib = gui.indentedBox(ribg)
        self.markInputCombo = gui.comboBox(ib, self, 'markInput',
                                           callback=lambda: self.set_selection_mode(SelectionMode.FROM_INPUT))
        self.markInputRadioButton.setEnabled(False)

        gui.auto_commit(ribg, self, 'do_auto_commit', 'Output changes')
        self.markTab.layout().addStretch(1)

        self.set_graph(None)
        self.set_selection_mode()

    def commit(self):
        self.send_data()

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
            self.tabs.widget(self.tabs.currentIndex()) != self.markTab):
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
                set(node for node, degree in self.graph.degree().items()
                    if degree >= self.markNConnections))
        elif selectionMode == SelectionMode.AT_MOST_N:
            self.view.setHighlighted(
                set(node for node, degree in self.graph.degree().items()
                    if degree <= self.markNConnections))
        elif selectionMode == SelectionMode.ANY_NEIGH:
            self.view.setHighlighted(
                set(node for node, degree in self.graph.degree().items()
                    if degree > max(self.graph.degree(self.graph[node]).values(), default=0)))
        elif selectionMode == SelectionMode.AVG_NEIGH:
            self.view.setHighlighted(
                set(node for node, degree in self.graph.degree().items()
                    if degree > np.nan_to_num(np.mean(list(self.graph.degree(self.graph[node]).values())))))
        elif selectionMode == SelectionMode.MOST_CONN:
            degrees = np.array(sorted(self.graph.degree().items(), key=lambda i: i[1], reverse=True))
            cut_ind = max(1, min(self.markNBest, self.graph.number_of_nodes()))
            cut_degree = degrees[cut_ind - 1, 1]
            toMark = set(degrees[degrees[:, 1] >= cut_degree, 0])
            self.view.setHighlighted(toMark)
        elif selectionMode == SelectionMode.FROM_INPUT:
            var = self.markInputCombo.currentText()
            tomark = {}
            if self.markInputItems:
                if var == 'ID':
                    values = {x.id for x in self.markInputItems}
                    tomark = {x for x in self.graph.nodes()
                              if self.graph.items()[x].id in values}
                else:
                    clean = lambda s: str(s).strip().upper()
                    values = {clean(x[var]) for x in self.markInputItems}
                    tomark = {x for x in self.graph.nodes()
                              if clean(self.graph.items()[x][var]) in values}
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
            for output in Output.all:
                self.send(output, None)
            return
        selected = self.view.getSelected()
        self.send(Output.SUBGRAPH,
                  self.graph.subgraph(selected) if selected else None)
        self.send(Output.DISTANCE,
                  self.items_matrix.submatrix(sorted(selected)) if self.items_matrix is not None and selected else None)
        items = self.graph.items()
        if not items:
            self.send(Output.SELECTED, None)
            self.send(Output.HIGHLIGHTED, None)
            self.send(Output.REMAINING, None)
        else:
            highlighted = self.view.getHighlighted()
            self.send(Output.SELECTED, items[sorted(selected), :] if selected else None)
            self.send(Output.HIGHLIGHTED, items[sorted(highlighted), :] if highlighted else None)
            remaining = sorted(set(self.graph) - set(selected) - set(highlighted))
            self.send(Output.REMAINING, items[remaining, :] if remaining else None)

    def _set_combos(self):
        self._clear_combos()
        self.graph_attrs = self.graph.items_vars()
        lastLabelColumns = self.lastLabelColumns
        lastTooltipColumns = self.lastTooltipColumns

        for var in self.graph_attrs:
            if var.is_discrete or var.is_continuous:
                self.colorCombo.addItem(gui.attributeIconDict[gui.vartype(var)], var.name, var)

            if var.is_continuous:
                self.nodeSizeCombo.addItem(gui.attributeIconDict[gui.vartype(var)], var.name, var)
            elif var.is_string:
                try: value = self.graph.items()[0][var].value
                except (IndexError, TypeError): pass
                else:
                    # can value be a list?
                    if len(value.split(',')) > 1:
                        self.nodeSizeCombo.addItem(gui.attributeIconDict[gui.vartype(var)], var.name, var)

        self.nodeSizeCombo.setDisabled(not self.graph_attrs)
        self.colorCombo.setDisabled(not self.graph_attrs)

        for i in range(self.nodeSizeCombo.count()):
            if self.lastVertexSizeColumn == \
                    self.nodeSizeCombo.itemText(i):
                self.node_size_attr = i
                self.set_node_sizes()
                break

        for i in range(self.colorCombo.count()):
            if self.lastColorColumn == self.colorCombo.itemText(i):
                self.node_color_attr = i
                self.set_node_colors()
                break

        for i in range(self.attListBox.count()):
            if str(self.attListBox.item(i).text()) in lastLabelColumns:
                self.attListBox.item(i).setSelected(True)
        self._on_node_label_attrs_changed()

        for i in range(self.tooltipListBox.count()):
            if (self.tooltipListBox.item(i).text() in lastTooltipColumns or
                not lastTooltipColumns):
                self.tooltipListBox.item(i).setSelected(True)

        self._clicked_tooltip_lstbox()

        self.lastLabelColumns = lastLabelColumns
        self.lastTooltipColumns = lastTooltipColumns

    def _clear_combos(self):
        self.graph_attrs = []

        self.colorCombo.clear()
        self.nodeSizeCombo.clear()

        self.colorCombo.addItem('(none)', None)
        self.nodeSizeCombo.addItem("(uniform)")

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

    def set_graph(self, graph):
        if not graph:
            return self.set_graph_none()
        if graph.number_of_nodes() < 2:
            self.set_graph_none()
            self.information('I\'m not really in a mood to visualize just one node. Try again tomorrow.')
            return
        self.information()

        all_edges_equal = bool(1 == len(set(w for u,v,w in graph.edges_iter(data='weight'))))
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
        if self.graph.number_of_nodes() + self.graph.number_of_edges() > 30000:
            self.set_graph_none()
            self.error('Network is too large to visualize. Sorry.')
            return
        self.error()

        self.set_selection_mode()
        self.relayout()

    def set_items(self, items=None):
        self._items = items
        if items is None:
            return self.set_graph(self.graph_base)
        if not self.graph:
            self.warning('No graph found!')
            return
        self.warning()
        if len(items) != self.graph.number_of_nodes():
            self.error('Items table must have one instance for each network node.')
            return
        self.error()
        self.graph.set_items(items)
        self._set_combos()

    def set_marking_items(self, items):
        self.markInputCombo.clear()
        self.markInputRadioButton.setEnabled(False)
        self.markInputItems = items

        self.warning()

        if items is None:
            return

        if self.graph is None or self.graph.items() is None:
            self.warning('No graph provided or no items attached to the graph.')
            return

        graph_items = self.graph.items()
        domain = graph_items.domain

        if len(items) > 0:
            commonVars = (set(x.name for x in chain(items.domain.variables,
                                                    items.domain.metas))
                          & set(x.name for x in chain(domain.variables,
                                                      domain.metas)))

            self.markInputCombo.addItem(gui.attributeIconDict[gui.vartype(DiscreteVariable())], "ID")

            for var in commonVars:
                orgVar, mrkVar = domain[var], items.domain[var]

                if type(orgVar) == type(mrkVar) == StringVariable:
                    self.markInputCombo.addItem(gui.attributeIconDict[gui.vartype(orgVar)], orgVar.name)

            self.markInputRadioButton.setEnabled(True)

    def relayout(self):
        if self.graph is None or self.graph.number_of_nodes() <= 1:
            return
        self.progressbar = gui.ProgressBar(self, FR_ITERATIONS)

        distmatrix = self.items_matrix
        if distmatrix is not None and distmatrix.shape[0] != self.graph.number_of_nodes():
            self.warning(17, "Distance matrix size doesn't match the number of network nodes. Not using it.")
            distmatrix = None
        self.warning(17)

        self.relayout_button.setDisabled(True)
        self.view.relayout(randomize=False, weight=distmatrix)

    def _on_node_label_attrs_changed(self):
        if not self.graph: return
        attributes = self.lastLabelColumns = [self.graph_attrs[i] for i in self.node_label_attrs]
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
        attributes = self.lastTooltipColumns = [self.graph_attrs[i] for i in self.tooltipAttributes]
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
        if self.showEdgeWeights:
            weights = (str(w or '') for u, v, w in self.graph.edges_iter(data='weight'))
        else:
            weights = ('' for i in range(self.graph.number_of_edges()))
        for edge, weight in zip(self.view.edges, weights):
            edge.setText(weight)

    def set_node_colors(self):
        if not self.graph: return
        self.lastColorColumn = self.colorCombo.currentText()
        attribute = self.colorCombo.itemData(self.colorCombo.currentIndex())
        assert not attribute or isinstance(attribute, Orange.data.Variable)
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
        elif attribute.is_discrete:
            DISCRETE_PALETTE = ColorPaletteGenerator(len(attribute.values))
            colors = DISCRETE_PALETTE[values]
        for node, color in zip(self.view.nodes, colors):
            node.setColor(color)

    def set_node_sizes(self):
        attribute = self.nodeSizeCombo.itemData(self.nodeSizeCombo.currentIndex())
        depending_widgets = (self.invertNodeSizeCheck, self.maxNodeSizeSpin)
        for w in depending_widgets:
            w.setDisabled(not attribute)
        if not self.graph: return
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
        else:
            for node in self.view.nodes:
                node.setSize(self.minNodeSize)
            return
        values = np.array(values)
        if self.invertNodeSize:
            values += np.nanmin(values) + 1
            values = 1/values
        nodemin, nodemax = np.nanmin(values), np.nanmax(values)
        if nodemin == nodemax:
            # np.polyfit borks on this condition
            sizes = (self.minNodeSize for i in range(len(self.view.nodes)))
        else:
            k, n = np.polyfit([nodemin, nodemax],
                              [self.minNodeSize, self.maxNodeSize], 1)
            sizes = values * k + n
            sizes[np.isnan(sizes)] = np.nanmean(sizes)
        for node, size in zip(self.view.nodes, sizes):
            node.setSize(size)

    def set_edge_sizes(self):
        if not self.graph: return
        if self.relativeEdgeWidths:
            widths = [self.graph.edge[u][v].get('weight', 1)
                      for u, v in self.graph.edges()]
            widths = scale(widths, .7, 8)
        else:
            widths = (.7 for i in range(self.graph.number_of_edges()))
        for edge, width in zip(self.view.edges, widths):
            edge.setSize(width)

    def sendReport(self):
        self.reportSettings("Graph data",
                            [("Number of vertices", self.graph.number_of_nodes()),
                             ("Number of edges", self.graph.number_of_edges()),
                             ("Vertices per edge", "%.3f" % self.verticesPerEdge),
                             ("Edges per vertex", "%.3f" % self.edgesPerVertex),
                             ])
        if self.node_color_attr or self.node_size_attr or self.node_label_attrs:
            self.reportSettings("Visual settings",
                                [self.node_color_attr and ("Vertex color", self.colorCombo.currentText()),
                                 self.node_size_attr and ("Vertex size", str(self.nodeSizeCombo.currentText()) + " (inverted)" if self.invertNodeSize else ""),
                                 self.node_label_attrs and ("Labels", ", ".join(self.graph_attrs[i].name for i in self.node_label_attrs)),
                                ])
        self.reportSection("Graph")
        self.reportImage(self.view)


if __name__ == "__main__":
    import sys
    a = QApplication(sys.argv)
    ow = OWNxExplorer()
    ow.show()

    def setNetwork(signal, data, id=None):
        if signal == 'Network':
            ow.set_graph(data)
        #if signal == 'Items':
        #    ow.set_items(data)

    import OWNxFile
    from os.path import join, dirname
    owFile = OWNxFile.OWNxFile()
    owFile.send = setNetwork
    owFile.openNetFile(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))
    #~ owFile.openFile(join(dirname(dirname(__file__)), 'networks', 'airtraffic.net'))
    #~ owFile.openFile(join(dirname(dirname(__file__)), 'networks', 'lastfm.net'))
    #~ owFile.show()
    #~ owFile.selectNetFile(0)

    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()
