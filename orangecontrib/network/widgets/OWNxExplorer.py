import os
from operator import itemgetter, add
from functools import reduce
from itertools import chain

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import Orange
from Orange.widgets import gui, widget, settings
from Orange.widgets.utils.colorpalette import ColorPaletteDlg
from Orange.widgets.unsupervised.owmds import torgerson as MDS

from Orange.data import Table, Domain, DiscreteVariable
import orangecontrib.network as network
from orangecontrib.network.widgets.OWNxCanvasQt import *


class Layout:
    NONE = 'None'
    FHR = 'Fruchterman-Reingold'
    FHR_WEIGHTED = 'Weighted Fruchterman-Reingold'
    RANDOM = 'Random'
    CIRCULAR = 'Circular'
    CONCENTRIC = 'Concentric'
    SPECTRAL = 'Spectral'
    FRAGVIZ = 'FragViz'
    MDS = 'Multi-dimensional scaling'
    PIVOT_MDS = 'Pivot MDS'
    all = (NONE,      FHR,        FHR_WEIGHTED,
           CIRCULAR,  CONCENTRIC, RANDOM,
           SPECTRAL)
    REQUIRES_DISTANCE_MATRIX = (FRAGVIZ, MDS, PIVOT_MDS)


class SelectionMode:
    NONE,       \
    SEARCH,     \
    NEIGHBORS,  \
    AT_LEAST_N, \
    AT_MOST_N,  \
    ANY_NEIGH,  \
    AVG_NEIGH,  \
    MOST_CONN,  \
    = range(8)  # FML


class Output:
    SUBGRAPH = 'Selected sub-network'
    DISTANCE = 'Distance matrix'
    SELECTED = 'Selected items'
    HIGHLIGHTED = 'Highlighted items'
    REMAINING = 'Remaining items'


class OWNxExplorer(widget.OWWidget):
    name = "Network Explorer"
    description = "Visually explore the network and its properties."
    icon = "icons/NetworkExplorer.svg"
    priority = 6420

    inputs = [("Network", network.Graph, "set_graph", widget.Default),
              ("Items", Table, "set_items"),
              ("Distances", Orange.misc.DistMatrix, "set_items_distance_matrix"),
              ("Net View", network.NxView, "set_network_view")]

    outputs = [(Output.SUBGRAPH, network.Graph),
               (Output.DISTANCE, Orange.misc.DistMatrix),
               (Output.SELECTED, Table),
               (Output.HIGHLIGHTED, Table),
               (Output.REMAINING, Table)]

    settingsList = ["autoSendSelection", "spinExplicit", "spinPercentage",
    "maxNodeSize", "invertNodeSize", "optMethod",
    "lastVertexSizeColumn", "lastColorColumn", "networkCanvas.show_indices", "networkCanvas.show_weights",
    "lastLabelColumns", "lastTooltipColumns",
    "showWeights", "colorSettings",
    "selectedSchemaIndex", "selectedEdgeSchemaIndex",
    "showMissingValues", "fontSize", "mdsTorgerson", "mdsAvgLinkage",
    "mdsSteps", "mdsRefresh", "mdsStressDelta", "showTextMiningInfo",
    "toolbarSelection", "minComponentEdgeWidth", "maxComponentEdgeWidth",
    "mdsFromCurrentPos", "labelsOnMarkedOnly", "tabIndex",
    "opt_from_curr",
    "fontWeight", "networkCanvas.state",
    "networkCanvas.selection_behavior", "hubs", "markDistance",
    "markNConnections", "markNumber", "markSearchString"]
    # TODO: set settings
    do_auto_commit = settings.Setting(True)

    def __init__(self):
        super().__init__()
        #self.contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", selected="node_label_attrs"), ContextField("attributes", selected="tooltipAttributes"), "color"])}

        self.networkCanvas = networkCanvas = OWNxCanvas(self)
        self.graph_attrs = []
        self.edges_attrs = []


        self.node_label_attrs = []
        self.tooltipAttributes = []
        self.edgeLabelAttributes = []
        self.autoSendSelection = False
        self.graphShowGrid = 1  # show gridlines in the graph
        self.markNConnections = 2
        self.markNumber = 0
        self.markProportion = 0
        self.markSearchString = ""
        self.markDistance = 2
        self.frSteps = 1
        self.hubs = 0
        self.node_color_attr = 0
        self.edgeColor = 0
        self.node_size_attr = 0
        self.nShown = self.nHidden = self.nHighlighted = self.nSelected = self.verticesPerEdge = self.edgesPerVertex = 0
        self.optimizeWhat = 1
        self.maxNodeSize = 50
        self.labelsOnMarkedOnly = 0
        self.invertNodeSize = 0
        self.optMethod = 0
        self.lastVertexSizeColumn = ''
        self.lastColorColumn = ''
        self.lastLabelColumns = set()
        self.lastTooltipColumns = set()
        self.showWeights = 0

        self.selectedEdgeSchemaIndex = 0
        self.items_matrix = None
        self.showDistances = 0
        self.showMissingValues = 0
        self.fontSize = 12
        self.fontWeight = 1
        self.mdsTorgerson = 0
        self.mdsAvgLinkage = 1
        self.mdsSteps = 10000
        self.mdsRefresh = 50
        self.mdsStressDelta = 0.0000001
        self.showTextMiningInfo = 0
        self.toolbarSelection = 0
        self.minComponentEdgeWidth = 10
        self.maxComponentEdgeWidth = 70
        self.mdsFromCurrentPos = 0
        self.tabIndex = 0
        self.number_of_nodes_label = -1
        self.number_of_edges_label = -1
        self.opt_from_curr = False

        self.checkSendMarkedNodes = True
        self.checkSendSelectedNodes = True
        self.marked_nodes = []

        self._network_view = None
        self.graph = None
        self.graph_base = None

        self.networkCanvas.showMissingValues = self.showMissingValues

        class ViewBox(pg.ViewBox):
            def __init__(self):
                super().__init__()

            def mouseDragEvent(self, ev):
                if not ev.isFinish():
                    return super().mouseDragEvent(ev)
                if self.state['mouseMode'] != self.RectMode:
                    return
                # Tap into pg.ViewBox's rbScaleBox ... it'll be fine.
                self.rbScaleBox.hide()
                ax = QRectF(pg.Point(ev.buttonDownPos(ev.button())),
                            pg.Point(ev.pos()))
                ax = self.childGroup.mapRectFromParent(ax)
                networkCanvas.selectNodesInRect(ax)
                self.setMouseMode(self.PanMode)
                ev.accept()

            def mouseClickEvent(self, ev):
                if ev.button() == Qt.LeftButton:
                    if networkCanvas.is_animating:
                        networkCanvas.is_animating = False
                        ev.accept()
                    else:
                        networkCanvas.mouseClickEvent(ev)
                super().mouseClickEvent(ev)

        class PlotItem(pg.PlotItem):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                for axis in ('bottom', 'left'):
                    self.hideAxis(axis)

                def _path(filename):
                    return os.path.join(os.path.dirname(__file__), 'icons', filename + '.png')

                self.rectBtn = pg.ButtonItem(_path('button_rect'), parentItem=self)
                self.rectBtn.clicked.connect(self.rectBtnClicked)

                self._qtBaseClass.setParentItem(self.autoBtn, None)
                self.autoBtn.hide()
                self.autoBtn = pg.ButtonItem(_path('button_autoscale'), parentItem=self)
                self.autoBtn.mode = 'auto'
                self.autoBtn.clicked.connect(self.autoBtnClicked)

                self.textEnterToSelect = pg.TextItem(
                    html='<div style="background-color:#f0f0f0; padding:5px;">'
                         '<font color="#444444"><b>Press <tt>Enter</tt> to add '
                         '<i><font color="{}">highlighted</font></i> nodes to '
                         '<i><font color="{}">selection</font></i> ...</font></b></div>'
                         .format(NodePenColor.HIGHLIGHTED, NodePenColor.SELECTED))
                self.textEnterToSelect.setParentItem(self)
                self.textEnterToSelect.hide()

            def rectBtnClicked(self, ev):
                self.vb.setMouseMode(self.vb.RectMode)

            def updateButtons(self):
                self.autoBtn.show()

            def resizeEvent(self, ev):
                super().resizeEvent(ev)
                btnRect = self.mapRectFromItem(self.rectBtn, self.rectBtn.boundingRect())
                LEFT_OFFSET, BOTTOM_OFFSET = 3, 5
                y = self.size().height() - btnRect.height() - BOTTOM_OFFSET
                self.autoBtn.setPos(LEFT_OFFSET, y)
                self.rectBtn.setPos(2*LEFT_OFFSET + btnRect.width(), y)
                self.textEnterToSelect.setPos(LEFT_OFFSET, BOTTOM_OFFSET)

        class PlotWidget(pg.PlotWidget):
            def __init__(self, *args, **kwargs):
                pg.GraphicsView.__init__(self, *args, **kwargs)

        plot = PlotWidget(self, background='w')
        plot.plotItem = PlotItem(enableAutoRange=True, viewBox=ViewBox())
        plot.setCentralItem(plot.plotItem)
        # Required, copied from pg.PlotWidget constructor
        for m in ['addItem', 'removeItem', 'autoRange', 'clear', 'setXRange',
                  'setYRange', 'setRange', 'setAspectLocked', 'setMouseEnabled',
                  'setXLink', 'setYLink', 'enableAutoRange', 'disableAutoRange',
                  'setLimits', 'register', 'unregister', 'viewRect']:
            setattr(plot, m, getattr(plot.plotItem, m))
        plot.plotItem.sigRangeChanged.connect(plot.viewRangeChanged)
        self.textEnterToSelect = plot.plotItem.textEnterToSelect

        plot.setFrameStyle(QFrame.StyledPanel)
        plot.setMinimumSize(500, 500)
        plot.setAspectLocked(True)
        plot.addItem(self.networkCanvas)
        self.mainArea.layout().addWidget(plot)

        self.tabs = gui.tabWidget(self.controlArea)

        self.displayTab = gui.createTabPage(self.tabs, "Display")
        self.markTab = gui.createTabPage(self.tabs, "Marking")

        def showTextOnMarkingTab(index):
            if self.tabs.widget(index) == self.markTab:
                self.set_mark_mode()
            else:
                self.acceptingEnterKeypress = False

        self.tabs.currentChanged.connect(showTextOnMarkingTab)

        self.tabs.setCurrentIndex(self.tabIndex)
        self.connect(self.tabs, SIGNAL("currentChanged(int)"), lambda index: setattr(self, 'tabIndex', index))

        ib = gui.widgetBox(self.displayTab, "Info")
        gui.label(ib, self, "Nodes: %(number_of_nodes_label)i (%(verticesPerEdge).2f per edge)")
        gui.label(ib, self, "Edges: %(number_of_edges_label)i (%(edgesPerVertex).2f per node)")

        box = gui.widgetBox(self.displayTab, "Nodes")
        self.optCombo = gui.comboBox(
            box, self, "optMethod", label='Layout:',
            orientation='horizontal', callback=self.graph_layout_method)
        for layout in Layout.all: self.optCombo.addItem(layout)
        self.optMethod = Layout.all.index(Layout.FHR)
        self.optCombo.setCurrentIndex(self.optMethod)

        self.colorCombo = gui.comboBox(
            box, self, "node_color_attr", label='Color by:',
            orientation='horizontal', callback=self.set_node_colors)

        hb = gui.widgetBox(box, orientation="horizontal", addSpace=False)
        hb.layout().addWidget(QLabel('Size by:', hb))
        self.nodeSizeCombo = gui.comboBox(
            hb, self, "node_size_attr", callback=self.set_node_sizes)
        self.maxNodeSizeSpin = gui.spin(
            hb, self, "maxNodeSize", 5, 200, step=10, label="Max:",
            callback=self.set_node_sizes)
        self.maxNodeSizeSpin.setValue(50)
        self.invertNodeSizeCheck = gui.checkBox(
            hb, self, "invertNodeSize", "Invert",
            callback=self.set_node_sizes)

        lb = gui.widgetBox(box, "Node labels | tooltips", orientation="vertical", addSpace=False)
        hb = gui.widgetBox(lb, orientation="horizontal", addSpace=False)
        self.attListBox = gui.listBox(
            hb, self, "node_label_attrs", "graph_attrs",
            selectionMode=QListWidget.MultiSelection,
            callback=self._on_node_label_attrs_changed)
        self.tooltipListBox = gui.listBox(
            hb, self, "tooltipAttributes", "graph_attrs",
            selectionMode=QListWidget.MultiSelection,
            callback=self._clicked_tooltip_lstbox)

        eb = gui.widgetBox(self.displayTab, "Edges", orientation="vertical")
        self.checkbox_relative_edges = gui.checkBox(
            eb, self, 'networkCanvas.relative_edge_widths', 'Relative edge widths',
            callback=self.networkCanvas.set_edge_sizes)
        self.checkbox_show_weights = gui.checkBox(
            eb, self, 'networkCanvas.show_edge_weights', 'Show edge weights',
            callback=self.networkCanvas.set_edge_labels)
        self.edgeColorCombo = gui.comboBox(
            eb, self, "edgeColor", label='Color by:', orientation='horizontal',
            callback=self.set_edge_colors)
        elb = gui.widgetBox(eb, "Edge labels", addSpace=False)
        self.edgeLabelListBox = gui.listBox(
            elb, self, "edgeLabelAttributes", "edges_attrs",
            selectionMode=QListWidget.MultiSelection,
            callback=self._clicked_edge_label_listbox)
        elb.setEnabled(False)


        ib = gui.widgetBox(self.markTab, "Info", orientation="vertical")
        gui.label(ib, self, "Nodes: %(number_of_nodes_label)i")
        gui.label(ib, self, "Selected: %(nSelected)i")
        gui.label(ib, self, "Highlighted: %(nHighlighted)i")

        ib = gui.widgetBox(self.markTab, "Highlight nodes ...")
        ribg = gui.radioButtonsInBox(ib, self, "hubs", [], "Mark", callback=self.set_mark_mode)
        gui.appendRadioButton(ribg, "None")
        gui.appendRadioButton(ribg, "... whose attributes contain:")
        self.ctrlMarkSearchString = gui.lineEdit(gui.indentedBox(ribg), self, "markSearchString", callback=self._set_search_string_timer, callbackOnType=True)
        self.searchStringTimer = QTimer(self)
        self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.set_mark_mode)

        gui.appendRadioButton(ribg, "... neighbours of selected, ≤ N hops away")
        ib = gui.indentedBox(ribg, orientation=0)
        self.ctrlMarkDistance = gui.spin(ib, self, "markDistance", 1, 100, 1, label="Distance:",
            callback=lambda: self.set_mark_mode(SelectionMode.NEIGHBORS))
        #self.ctrlMarkFreeze = gui.button(ib, self, "&Freeze", value="graph.freezeNeighbours", toggleButton = True)
        gui.appendRadioButton(ribg, "... with at least N connections")
        gui.appendRadioButton(ribg, "... with at most N connections")
        self.ctrlMarkNConnections = gui.spin(gui.indentedBox(ribg), self, "markNConnections", 0, 1000000, 1, label="N:",
            callback=lambda: self.set_mark_mode(SelectionMode.AT_MOST_N if self.hubs == SelectionMode.AT_MOST_N else SelectionMode.AT_LEAST_N))
        gui.appendRadioButton(ribg, "... with more connections than any neighbor")
        gui.appendRadioButton(ribg, "... with more connections than average neighbor")
        gui.appendRadioButton(ribg, "... with most connections")
        ib = gui.indentedBox(ribg)
        #~ self.ctrlMarkNumber = gui.spin(ib, self, "markNumber", 1, 1000000, 1, label="Number of nodes:", callback=(lambda h=7: self.set_mark_mode(h)))
        self.ctrlMarkNumber = gui.spin(ib, self, "markNumber", 1, 1000000, 1, label="Number of nodes:", callback=lambda: self.set_mark_mode(SelectionMode.MOST_CONN))

        gui.auto_commit(ribg, self, 'do_auto_commit', 'Output changes')

        #ib = gui.widgetBox(self.markTab, "General", orientation="vertical")
        #self.checkSendMarkedNodes = True

        self.toolbar = gui.widgetBox(self.controlArea, orientation='horizontal')
        #~ G = self.networkCanvas.gui
        #~ self.zoomSelectToolbar = G.zoom_select_toolbar(self.toolbar, nomargin=True, buttons=
            #~ G.default_zoom_select_buttons +
            #~ [
                #~ G.Spacing,
                #~ ("buttonM2S", "Marked to selection", None, None, "marked_to_selected", 'Dlg_Mark2Sel'),
                #~ ("buttonS2M", "Selection to marked", None, None, "selected_to_marked", 'Dlg_Sel2Mark'),
                #~ ("buttonHSEL", "Hide selection", None, None, "hide_selection", 'Dlg_HideSelection'),
                #~ ("buttonSSEL", "Show all nodes", None, None, "show_selection", 'Dlg_ShowSelection'),
                #~ #("buttonUN", "Hide unselected", None, None, "hideUnSelectedVertices", 'Dlg_SelectedNodes'),
                #~ #("buttonSW", "Show all nodes", None, None, "showAllVertices", 'Dlg_clear'),
            #~ ])
        #~ self.zoomSelectToolbar.buttons[G.SendSelection].clicked.connect(self.send_data)
        #~ self.zoomSelectToolbar.buttons[G.SendSelection].clicked.connect(self.send_marked_nodes)
        #~ self.zoomSelectToolbar.buttons[("buttonHSEL", "Hide selection", None, None, "hide_selection", 'Dlg_HideSelection')].clicked.connect(self.hide_selection)
        #~ self.zoomSelectToolbar.buttons[("buttonSSEL", "Show all nodes", None, None, "show_selection", 'Dlg_ShowSelection')].clicked.connect(self.show_selection)
        #self.zoomSelectToolbar.buttons[G.SendSelection].hide()

        self.set_mark_mode()

        self.displayTab.layout().addStretch(1)
        self.markTab.layout().addStretch(1)

        self.graph_layout_method()
        self.set_graph(None)

        self.setMinimumWidth(900)

    def commit(self):
        self.send_data()

    def hide_selection(self):
        nodes = set(self.graph.nodes()).difference(self.networkCanvas.selected_nodes())
        self.change_graph(network.nx.Graph.subgraph(self.graph, nodes))

    def show_selection(self):
        self.change_graph(self.graph_base)

    def edit(self):
        if self.graph is None:
            return

        vars = [x.name for x in self.graph_base.items_vars()]
        vertices = self.networkCanvas.selected_nodes()

        if len(vertices) == 0:
            return

        items = self.graph_base.items()
        if items.domain[att].is_continuous:
            for v in vertices:
                items[v][att] = float(self.editValue)
        else:
            for v in vertices:
                items[v][att] = str(self.editValue)

    def set_items_distance_matrix(self, matrix):
        assert matrix is None or isinstance(matrix, Orange.misc.DistMatrix)
        self.error()
        self.warning()
        self.information()

        self.items_matrix = matrix

        if matrix is None:
            return

        if self.graph_base is None:
            self.networkCanvas.items_matrix = None
            self.information('No graph found!')
            return

        if matrix.dim != self.graph_base.number_of_nodes():
            self.error('The number of vertices does not match matrix size.')
            self.items_matrix = None
            self.networkCanvas.items_matrix = None
            return

        self.networkCanvas.items_matrix = matrix

        if Layout.all[self.optMethod] in Layout.REQUIRES_DISTANCE_MATRIX:
            if self.items_matrix is not None and self.graph_base is not None and \
                                self.items_matrix.dim == self.graph_base.number_of_nodes():

                if self.optMethod == Layout.FRAGVIZ: # if FragViz, run FR first
                    self.optMethod = Layout.all.index(Layout.FHR)
                    self.graph_layout()
                    self.optMethod = Layout.all.index(Layout.FRAGVIZ)

            self.graph_layout()

    def _set_curve_attr(self, attr, value):
        setattr(self.networkCanvas.networkCurve, attr, value)
        self.networkCanvas.updateCanvas()

    def _set_search_string_timer(self):
        self.hubs = 1
        self.searchStringTimer.stop()
        self.searchStringTimer.start(300)

    @property
    def acceptingEnterKeypress(self):
        return self.textEnterToSelect.isVisible()

    @acceptingEnterKeypress.setter
    def acceptingEnterKeypress(self, v):
        if v: self.textEnterToSelect.show()
        else: self.textEnterToSelect.hide()

    def set_mark_mode(self, i=None):
        self.searchStringTimer.stop()
        hubs = self.hubs = i or self.hubs
        if (self.graph is None or
            self.tabs.widget(self.tabs.currentIndex()) != self.markTab):
            return

        self.acceptingEnterKeypress = True

        if hubs == SelectionMode.NONE:
            self.networkCanvas.setHighlighted([])
            self.acceptingEnterKeypress = False
        elif hubs == SelectionMode.SEARCH:
            table, txt = self.graph.items(), self.markSearchString.lower()
            if not table or not txt: return
            toMark = set(i for i, instance in enumerate(table)
                         if txt in " ".join(map(str, instance.list)).lower())
            self.networkCanvas.setHighlighted(toMark)
        elif hubs == SelectionMode.NEIGHBORS:
            neighbors = set(self.networkCanvas.selectedNodes)
            for _ in range(self.markDistance):
                for neigh in list(neighbors):
                    neighbors |= set(self.graph[neigh].keys())
            neighbors -= self.networkCanvas.selectedNodes
            self.networkCanvas.setHighlighted(neighbors)
        elif hubs == SelectionMode.AT_LEAST_N:
            self.networkCanvas.setHighlighted(
                set(node for node, degree in self.graph.degree().items()
                    if degree >= self.markNConnections))
        elif hubs == SelectionMode.AT_MOST_N:
            self.networkCanvas.setHighlighted(
                set(node for node, degree in self.graph.degree().items()
                    if degree <= self.markNConnections))
        elif hubs == SelectionMode.ANY_NEIGH:
            self.networkCanvas.setHighlighted(
                set(node for node, degree in self.graph.degree().items()
                    if degree > max(self.graph.degree(self.graph[node]).values(), default=0)))
        elif hubs == SelectionMode.AVG_NEIGH:
            self.networkCanvas.setHighlighted(
                set(node for node, degree in self.graph.degree().items()
                    if degree > np.nan_to_num(np.mean(list(self.graph.degree(self.graph[node]).values())))))
        elif hubs == SelectionMode.MOST_CONN:
            degrees = np.array(sorted(self.graph.degree().items(), key=lambda i: i[1], reverse=True))
            cut_ind = max(1, min(self.markNumber, self.graph.number_of_nodes()))
            cut_degree = degrees[cut_ind - 1, 1]
            toMark = set(degrees[degrees[:, 1] >= cut_degree, 0])
            self.networkCanvas.setHighlighted(toMark)

    def keyReleaseEvent(self, ev):
        """On Enter, expand the selected set with the highlighted"""
        if (not self.acceptingEnterKeypress or
            ev.key() not in (Qt.Key_Return, Qt.Key_Enter)):
            super().keyReleaseEvent(ev)
            return
        self.networkCanvas.selectHighlighted()
        self.set_mark_mode()

    def save_network(self):
        if self.networkCanvas is None or self.graph is None:
            return

        filename = QFileDialog.getSaveFileName(self, 'Save Network File', \
            '', 'NetworkX graph as Python pickle (*.gpickle)\nPajek ' + \
            'network (*.net)\nGML network (*.gml)')
        filename = str(filename)
        if filename:
            fn = ""
            head, tail = os.path.splitext(filename)
            if not tail:
                fn = head + ".net"
            else:
                fn = filename

            items = self.graph.items()
            for i in range(self.graph.number_of_nodes()):
                graph_node = self.graph.node[i]
                plot_node = self.networkCanvas.networkCurve.nodes()[i]

                if items is not None:
                    ex = items[i]
                    if 'x' in ex.domain:
                        ex['x'] = plot_node.x()
                    if 'y' in ex.domain:
                        ex['y'] = plot_node.y()

                graph_node['x'] = plot_node.x()
                graph_node['y'] = plot_node.y()

            network.readwrite.write(self.graph, fn)

    def send_data(self):
        selected = self.networkCanvas.selectedNodes
        highlighted = self.networkCanvas.highlightedNodes

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
            self.send(Output.SELECTED, items[sorted(selected), :] if selected else None)
            self.send(Output.HIGHLIGHTED, items[sorted(highlighted), :] if highlighted else None)
            remaining = sorted(set(self.graph) - selected - highlighted)
            self.send(Output.REMAINING, items[remaining, :] if remaining else None)

    def _set_combos(self):
        self._clear_combos()
        self.graph_attrs = self.graph_base.items_vars()
        self.edges_attrs = self.graph_base.links_vars()
        lastLabelColumns = self.lastLabelColumns
        lastTooltipColumns = self.lastTooltipColumns

        for var in self.graph_attrs:
            if (var.is_discrete or
                var.is_continuous or
                var.is_string and var.name == 'label'):  # FIXME: whatis label?
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

        for var in self.edges_attrs:
            if var.is_discrete or var.is_continuous:
                self.edgeColorCombo.addItem(gui.attributeIconDict[gui.vartype(var)], str(var.name))
        self.edgeColorCombo.setDisabled(not self.edges_attrs)

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
        self.edges_attrs = []

        self.colorCombo.clear()
        self.nodeSizeCombo.clear()
        self.edgeColorCombo.clear()

        self.colorCombo.addItem('(none)', None)
        self.edgeColorCombo.addItem("(same color)")
        self.nodeSizeCombo.addItem("(uniform)")

    def compute_network_info(self):
        self.nShown = self.graph.number_of_nodes()

        if self.graph.number_of_edges() > 0:
            self.verticesPerEdge = float(self.graph.number_of_nodes()) / float(self.graph.number_of_edges())
        else:
            self.verticesPerEdge = 0

        if self.graph.number_of_nodes() > 0:
            self.edgesPerVertex = float(self.graph.number_of_edges()) / float(self.graph.number_of_nodes())
        else:
            self.edgesPerVertex = 0

    def change_graph(self, newgraph):
        self.information()

        # if graph has more nodes and edges than pixels in 1600x1200 display,
        # it is too big to visualize!
        if newgraph.number_of_nodes() + newgraph.number_of_edges() > 50000:
            self.information('New graph is too big to visualize. Keeping the old graph.')
            return

        self.graph = newgraph

        self.number_of_nodes_label = self.graph.number_of_nodes()
        self.number_of_edges_label = self.graph.number_of_edges()

        if not self.networkCanvas.change_graph(self.graph):
            return

        self.compute_network_info()

        if self.graph.number_of_nodes() > 0:
            t = 1.13850193174e-008 * (self.graph.number_of_nodes() ** 2 + self.graph.number_of_edges())
            self.frSteps = int(2.0 / t)
            if self.frSteps < 1: self.frSteps = 1
            if self.frSteps > 100: self.frSteps = 100
#
#        if self.frSteps < 10:
#            self.networkCanvas.use_antialiasing = 0
#            self.networkCanvas.use_animations = 0
#            self.minVertexSize = 5
#            self.maxNodeSize = 5
#            self.maxLinkSize = 1
#            self.optMethod = 0
#            self.graph_layout_method()

        animation_enabled = self.networkCanvas.animate_points;
        self.networkCanvas.animate_points = False;

        self.set_node_sizes()
        self.set_node_colors()
        self.set_edge_colors()

        self._on_node_label_attrs_changed()
        self._clicked_tooltip_lstbox()
        self._clicked_edge_label_listbox()

        self.networkCanvas.replot()

        self.networkCanvas.animate_points = animation_enabled
        qApp.processEvents()
        self.networkCanvas.networkCurve.layout_fr(100, weighted=False, smooth_cooling=True)
        self.networkCanvas.replot()

    def set_graph_none(self):
        self.graph = None
        self.graph_base = None
        #self.graph_base = None
        self._clear_combos()
        self.number_of_nodes_label = -1
        self.number_of_edges_label = -1
        self.verticesPerEdge = -1
        self.edgesPerVertex = -1
        self._items = None
        self._links = None
        self.set_items_distance_matrix(None)
        self.networkCanvas.set_graph(None)

    def set_graph(self, graph):
        self.information()
        self.warning()
        self.error()

        if graph is None:
            self.set_graph_none()
            return

        all_edges_equal = bool(1 == len(set(w for u,v,w in graph.edges_iter(data='weight'))))
        self.checkbox_show_weights.setEnabled(not all_edges_equal)
        self.checkbox_relative_edges.setEnabled(not all_edges_equal)
        self.optCombo.model().item(0).setEnabled(bool(graph.items()))

        if graph.number_of_nodes() < 2:
            self.set_graph_none()
            self.information('I\'m not really in a mood to visualize just one node. Try again tomorrow.')
            return

        if graph == self.graph_base and self.graph is not None and \
                                                self._network_view is None:
            self.set_items(graph.items())
            return

        if self._network_view is not None:
            graph = self._network_view.init_network(graph)

        self.graph = self.graph_base = graph

        # if graph has more nodes and edges than pixels in 1600x1200 display,
        # it is too big to visualize!
        if self.graph.number_of_nodes() + self.graph.number_of_edges() > 50000:
            self.set_graph_none()
            self.error('Graph is too big to visualize. Try using one of the network views.')
            return

        if self.items_matrix is not None and self.items_matrix.dim != self.graph_base.number_of_nodes():
            self.set_items_distance_matrix(None)

        self.number_of_nodes_label = self.graph.number_of_nodes()
        self.number_of_edges_label = self.graph.number_of_edges()

        self.networkCanvas.minComponentEdgeWidth = self.minComponentEdgeWidth
        self.networkCanvas.maxComponentEdgeWidth = self.maxComponentEdgeWidth
        #~ self.networkCanvas.set_labels_on_marked(self.labelsOnMarkedOnly)

        self._set_combos()
        self.compute_network_info()

        t = 1.13850193174e-008 * (self.graph.number_of_nodes() ** 2 + self.graph.number_of_edges())
        self.frSteps = int(2.0 / t)
        if self.frSteps < 1: self.frSteps = 1
        if self.frSteps > 100: self.frSteps = 100

        self.networkCanvas.set_antialias(self.graph.number_of_nodes() +
                                         self.graph.number_of_edges() < 1000)
        # if graph is large, set random layout, min vertex size, min edge size
        if self.frSteps < 10:
            self.minVertexSize = 5
            self.maxNodeSize = 5
            #~ self.optMethod = 0
            self.graph_layout_method()

        self.networkCanvas.labelsOnMarkedOnly = self.labelsOnMarkedOnly
        self.networkCanvas.showWeights = self.showWeights

        self.networkCanvas.set_graph(self.graph)
        self.set_node_sizes()
        self.set_node_colors()
        self.set_edge_colors()

        self._on_node_label_attrs_changed()
        self._clicked_tooltip_lstbox()
        self._clicked_edge_label_listbox()

        self.set_mark_mode()

    def set_network_view(self, nxView):
        self.error()
        self.warning()
        self.information()

        if self.graph is None:
            self.information('Do not forget to add a graph!')

        if self._network_view is not None:
            QObject.disconnect(self.networkCanvas, SIGNAL('selection_changed()'), self._network_view.node_selection_changed)

        self._network_view = nxView

        g = self.graph_base
        if self._network_view is not None:
            self._network_view.set_nx_explorer(self)
        else:
            self.graph_base = None

        self.set_graph(g)

        if self._network_view is not None:
            QObject.connect(self.networkCanvas, SIGNAL('selection_changed()'), self._network_view.node_selection_changed)

    def set_items(self, items=None):
        self.error()
        self.warning()
        self.information()

        if items is None:
            return

        if self.graph is None:
            self.warning('No graph found!')
            return

        if len(items) != self.graph_base.number_of_nodes():
            self.error('Table items must have one example for each node.')
            return

        self.graph_base.set_items(items)

        self.set_node_sizes()
        self.networkCanvas.items = items
        self.networkCanvas.showWeights = self.showWeights
        self._set_combos()
        #self.networkCanvas.updateData()

    def explore_focused(self):
        sel = self.networkCanvas.selected_nodes()
        if len(sel) == 1:
            ndx_1 = sel[0]
            self.networkCanvas.label_distances = [['%.2f' % \
                            self.items_matrix[ndx_1][ndx_2]] \
                            for ndx_2 in self.networkCanvas.graph.nodes()]
        else:
            self.networkCanvas.label_distances = None

        self.networkCanvas.set_node_labels(self.lastLabelColumns)
        self.networkCanvas.replot()

    #######################################################################
    ### Layout Optimization                                             ###
    #######################################################################

    def graph_layout(self):
        if self.graph is None or self.graph.number_of_nodes() <= 0:   #grafa se ni
            return

        # Cancel previous animation if running
        self.networkCanvas.is_animating = False

        layout = Layout.all[self.optMethod]
        if layout == Layout.NONE:
            self.networkCanvas.layout_original()
        elif layout == Layout.RANDOM:
            self.networkCanvas.layout_random()
        elif layout == Layout.FHR:
            self.networkCanvas.layout_fhr(False)
        elif layout == Layout.FHR_WEIGHTED:
            self.networkCanvas.layout_fhr(True)
        elif layout == Layout.CONCENTRIC:
            self.networkCanvas.layout_concentric()
        elif layout == Layout.CIRCULAR:
            self.networkCanvas.layout_circular()
        elif layout == Layout.SPECTRAL:
            self.networkCanvas.layout_spectral()
        elif layout == Layout.FRAGVIZ:
            self.graph_layout_fragviz()
        elif layout == Layout.MDS:
            self.graph_layout_mds()
        elif layout == Layout.PIVOT_MDS:
            self.graph_layout_pivot_mds()
        else: raise Exception('wtf')
        self.networkCanvas.replot()

    def graph_layout_method(self):
        self.information()

        if Layout.all[self.optMethod] in Layout.REQUIRES_DISTANCE_MATRIX:
            if self.items_matrix is None:
                self.information('Set distance matrix to input signal')
                return
            if self.graph is None:
                self.information('No network found')
                return
            if self.items_matrix.dim != self.graph_base.number_of_nodes():
                self.error('Distance matrix dimensionality must equal number of vertices')
                return

        self.graph_layout()

    def mds_progress(self, avgStress, stepCount):
        #self.drawForce()

        #self.mdsInfoA.setText("Avg. Stress: %.20f" % avgStress)
        #self.mdsInfoB.setText("Num. steps: %i" % stepCount)
        self.progressBarSet(int(stepCount * 100 / self.frSteps))
        qApp.processEvents()

    def graph_layout_fragviz(self):
        if self.items_matrix is None:
            self.information('Set distance matrix to input signal')
            return

        if self.layout is None:
            self.information('No network found')
            return

        if self.items_matrix.dim != self.graph_base.number_of_nodes():
            self.error('Distance matrix dimensionality must equal number of vertices')
            return

        self.progressBarInit()
        qApp.processEvents()

        if self.graph.number_of_nodes() == self.graph_base.number_of_nodes():
            matrix = self.items_matrix
        else:
            matrix = self.items_matrix.get_items(sorted(self.graph.nodes_iter()))

        self.networkCanvas.networkCurve.layout_fragviz(self.frSteps, matrix, self.graph, self.mds_progress, self.opt_from_curr)

        self.progressBarFinished()

    def graph_layout_mds(self):
        if self.items_matrix is None:
            self.information('Set distance matrix to input signal')
            return

        if self.layout is None:
            self.information('No network found')
            return

        if self.items_matrix.dim != self.graph_base.number_of_nodes():
            self.error('Distance matrix dimensionality must equal number of vertices')
            return

        self.progressBarInit()
        qApp.processEvents()

        if self.graph.number_of_nodes() == self.graph_base.number_of_nodes():
            matrix = self.items_matrix
        else:
            matrix = self.items_matrix.get_items(sorted(self.graph.nodes()))

        self.networkCanvas.networkCurve.layout_mds(self.frSteps, matrix, self.mds_progress, self.opt_from_curr)

        self.progressBarFinished()

    def graph_layout_pivot_mds(self):
        self.information()

        if self.items_matrix is None:
            self.information('Set distance matrix to input signal')
            return

        if self.graph_base is None:
            self.information('No network found')
            return

        if self.items_matrix.dim != self.graph_base.number_of_nodes():
            self.error('The number of vertices does not match matrix size.')
            return

        self.frSteps = min(self.frSteps, self.graph.number_of_nodes())
        qApp.processEvents()

        if self.graph.number_of_nodes() == self.graph_base.number_of_nodes():
            matrix = self.items_matrix
        else:
            matrix = self.items_matrix.get_items(sorted(self.graph.nodes()))

        mds = MDS(matrix, self.frSteps)
        x, y = mds.optimize()
        xy = zip(list(x), list(y))
        coors = dict(zip(sorted(self.graph.nodes()), xy))
        self.networkCanvas.networkCurve.set_node_coordinates(coors)
        self.networkCanvas.update_layout()

    #######################################################################
    ### Network Visualization                                           ###
    #######################################################################

    def _on_node_label_attrs_changed(self):
        if self.graph is None:
            return
        self.lastLabelColumns = [self.graph_attrs[i] for i in self.node_label_attrs]  # TODO
        self.networkCanvas.set_node_labels(self.lastLabelColumns)

    def _clicked_tooltip_lstbox(self):
        if self.graph is None:
            return
        self.lastTooltipColumns = [self.graph_attrs[i] for i in self.tooltipAttributes]
        self.networkCanvas.set_tooltip_attributes(self.lastTooltipColumns)

    def _clicked_edge_label_listbox(self):
        self.lastEdgeLabelAttributes = [self.edges_attrs[i] for i in self.edgeLabelAttributes]
        self.networkCanvas.set_edge_labels(self.lastEdgeLabelAttributes)

    def set_node_colors(self):
        self.networkCanvas.set_node_colors(self.colorCombo.itemData(self.colorCombo.currentIndex()))
        self.lastColorColumn = self.colorCombo.currentText()  # TODO

    def set_edge_colors(self):
        self.networkCanvas.set_edge_colors(self.edgeColorCombo.itemData(self.edgeColorCombo.currentIndex()))
        self.lastEdgeColorColumn = self.edgeColorCombo.currentText()

    def set_node_sizes(self):
        attr = self.nodeSizeCombo.itemData(self.nodeSizeCombo.currentIndex())
        depending_widgets = (self.invertNodeSizeCheck, self.maxNodeSizeSpin)
        for w in depending_widgets:
            w.setDisabled(not bool(attr))
        self.networkCanvas.set_node_sizes(attr, self.maxNodeSize, self.invertNodeSize)

    def set_font(self):
        if self.networkCanvas is None:
            return

        weights = {0: 50, 1: 80}

        #~ font = self.networkCanvas.font()
        #~ font.setPointSize(self.fontSize)
        #~ font.setWeight(weights[self.fontWeight])
        #~ self.networkCanvas.setFont(font)
        #~ self.networkCanvas.fontSize = font
        #~ self.networkCanvas.set_node_labels()

    def sendReport(self):
        self.reportSettings("Graph data",
                            [("Number of vertices", self.graph.number_of_nodes()),
                             ("Number of edges", self.graph.number_of_edges()),
                             ("Vertices per edge", "%.3f" % self.verticesPerEdge),
                             ("Edges per vertex", "%.3f" % self.edgesPerVertex),
                             ])
        if self.node_color_attr or self.node_size_attr or self.node_label_attrs or self.edgeColor:
            self.reportSettings("Visual settings",
                                [self.node_color_attr and ("Vertex color", self.colorCombo.currentText()),
                                 self.node_size_attr and ("Vertex size", str(self.nodeSizeCombo.currentText()) + " (inverted)" if self.invertNodeSize else ""),
                                 self.node_label_attrs and ("Labels", ", ".join(self.graph_attrs[i].name for i in self.node_label_attrs)),
                                 self.edgeColor and ("Edge colors", self.edgeColorCombo.currentText()),
                                ])
        self.reportSettings("Optimization",
                            [("Method", self.optCombo.currentText()),
                             ("Iterations", self.frSteps)])
        self.reportSection("Graph")
        self.reportImage(self.networkCanvas.saveToFileDirect)


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
    owFile.openFile(join(dirname(dirname(__file__)), 'networks', 'leu_by_genesets.net'))
    #~ owFile.show()
    #~ owFile.selectNetFile(0)

    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()
