#
# OWHist.py
#
# the base for network histograms

import math
from itertools import chain
import numpy as np

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import Orange
from Orange.widgets import gui, widget
import orangecontrib.network as network

import pyqtgraph as pg


class OWNxHist():

    def __init__(self, parent=None, type=0):
        self.parent = parent

        # set default settings
        self.spinLowerThreshold = 0
        self.spinLowerChecked = False
        self.spinUpperThreshold = 0
        self.spinUpperChecked = False
        self.netOption = 0
        self.dstWeight = 0
        self.kNN = 0
        self.andor = 0
        self.matrix = None
        self.excludeLimit = 2
        self.percentil = 0

        self.graph = None
        self.graph_matrix = None

    def addHistogramControls(self, parent=None):
        if parent is None:
            parent = self.controlArea

        boxGeneral = gui.widgetBox(parent, box="Edges")
        ribg = gui.widgetBox(boxGeneral, None, orientation="horizontal", addSpace=False)
        ribg.layout().addWidget(QLabel("Distance threshold", self),
                                4, Qt.AlignVCenter | Qt.AlignLeft)
        self.spin_low = gui.doubleSpin(ribg, self, "spinLowerThreshold",
                         0.0, float("inf"), 0.001, decimals=3,
                         callback=self.changeLowerSpin,
                         keyboardTracking=False)
        ribg.layout().addWidget(QLabel("to", self), 1, Qt.AlignCenter)
        self.spin_high = gui.doubleSpin(ribg, self, "spinUpperThreshold",
                         0.0, float("inf"), 0.001, decimals=3,
                         callback=self.changeUpperSpin,
                         keyboardTracking=False)
        self.histogram.region.sigRegionChangeFinished.connect(self.spinboxFromHistogramRegion)
#         gui.lineEdit(ribg, self, "spinLowerThreshold", "Distance threshold   ", orientation='horizontal', callback=self.changeLowerSpin, valueType=float, validator=self.validator, enterPlaceholder=True, controlWidth=60)
#         gui.lineEdit(ribg, self, "spinUpperThreshold", "", orientation='horizontal', callback=self.changeUpperSpin, valueType=float, validator=self.validator, enterPlaceholder=True, controlWidth=60)
#         ribg.layout().addStretch(1)
        #ribg = gui.radioButtonsInBox(boxGeneral, self, "andor", [], orientation='horizontal', callback = self.generateGraph)
        #gui.appendRadioButton(ribg, self, "andor", "OR", callback = self.generateGraph)
        #b = gui.appendRadioButton(ribg, self, "andor", "AND", callback = self.generateGraph)
        #b.setEnabled(False)
        #ribg.hide(False)

        ribg = gui.widgetBox(boxGeneral, None, orientation="horizontal", addSpace=False)

        gui.doubleSpin(boxGeneral, self, "percentil", 0, 100, 0.1, label="Percentile", orientation='horizontal', callback=self.setPercentil, callbackOnReturn=1, controlWidth=60)
        gui.spin(boxGeneral, self, "kNN", 0, 1000, 1, label="Include closest neighbors", orientation='horizontal', callback=self.generateGraph, callbackOnReturn=1, controlWidth=60)
        ribg.layout().addStretch(1)
        # Options
        self.attrColor = ""
        ribg = gui.radioButtonsInBox(parent, self, "netOption", [], "Node selection", callback=self.generateGraph)
        gui.appendRadioButton(ribg, "Keep all nodes")
        hb = gui.widgetBox(ribg, None, orientation="horizontal", addSpace=False)
        gui.appendRadioButton(ribg, "Components with at least nodes", insertInto=hb)
        gui.spin(hb, self, "excludeLimit", 2, 100, 1, callback=(lambda h=True: self.generateGraph(h)), controlWidth=60)
        gui.appendRadioButton(ribg, "Largest connected component")
        #gui.appendRadioButton(ribg, self, "netOption", "Connected component with vertex")
        self.attribute = None

        ### FILTER NETWORK BY ATTRIBUTE IS OBSOLETE - USE SELECT DATA WIDGET ###
        #self.attributeCombo = gui.comboBox(parent, self, "attribute", box="Filter attribute", orientation='horizontal')#, callback=self.setVertexColor)
        #self.label = ''
        #self.searchString = gui.lineEdit(self.attributeCombo.box, self, "label", callback=self.setSearchStringTimer, callbackOnType=True)
        #self.searchStringTimer = QTimer(self)
        #self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.generateGraph)
        #if str(self.netOption) != '3':
        #    self.attributeCombo.box.setEnabled(False)

        ribg = gui.radioButtonsInBox(parent, self, "dstWeight", [], "Edge weights", callback=self.generateGraph)
        hb = gui.widgetBox(ribg, None, orientation="horizontal", addSpace=False)
        gui.appendRadioButton(ribg, "Proportional to distance", insertInto=hb)
        gui.appendRadioButton(ribg, "Inverted distance", insertInto=hb)

    def setPercentil(self):
        if self.matrix is None or self.percentil <= 0:
            return

        self.spinLowerThreshold = self.histogram.boundary()[0]
        # flatten matrix, sort values and remove identities (self.matrix[i][i])
        vals = sorted(self.matrix.flat)
        ind = int(len(vals) * self.percentil / 100)
        self.spinUpperThreshold = vals[ind]
        self.generateGraph()

    def setMatrix(self, data):
        if data is None:
            self.matrix = None
            self.histogram.setValues([])
            self.generateGraph()
            return

        if not hasattr(data, "items") or data.items is None:
            setattr(data, "items", [i for i in range(data.shape[0])])

        self.matrix = data
        # draw histogram
        values = data.flat
        # print("values:", values)
        self.histogram.setValues(values)

        # Magnitude of the spinbox's step is data-dependent
        step = round((values.std() / 5), -1)
        self.spin_low.setSingleStep(step)
        self.spin_high.setSingleStep(step)

        low = min(values)
        upp = max(values)

        self.spinLowerThreshold = self.spinUpperThreshold = low - (0.03 * (upp - low))

        # self.attributeCombo.clear()
        vars = []

        if hasattr(self.matrix, "items"):

            if isinstance(self.matrix.items, Orange.data.Table):
                vars = list(self.matrix.items.domain.variables)

                metas = self.matrix.items.domain.getmetas(0)
                for i, var in metas.items():
                    vars.append(var)

        self.icons = gui.attributeIconDict

        # for var in vars:
        #     try:
        #         if var.varType != 7: # if not Orange.feature.Python
        #             self.attributeCombo.addItem(self.icons[var.varType], unicode(var.name))
        #     except:
        #         print "Error adding", var, "to the attribute combo."

        self.setPercentil()
        self.generateGraph()

    def changeUpperSpin(self):
        if self.spinLowerThreshold > self.spinUpperThreshold:
            self.spinLowerThreshold = self.spinUpperThreshold
        self.changeLowerSpin()

    def changeLowerSpin(self):
        self.percentil = 0
        self.spinLowerThreshold, self.spinUpperThreshold = np.clip(
            [self.spinLowerThreshold, self.spinUpperThreshold],
            *self.histogram.boundary())
        if self.spinLowerThreshold > self.spinUpperThreshold:
            self.spinUpperThreshold = self.spinLowerThreshold
        self.generateGraph()

    def spinboxFromHistogramRegion(self):
        self.spinLowerThreshold, self.spinUpperThreshold = self.histogram.getRegion()

    def generateGraph(self, N_changed=False):
        self.error()
        matrix = None
        self.warning('')

        if N_changed:
            self.netOption = 1

        if self.matrix is None:
            if hasattr(self, "infoa"):
                self.infoa.setText("No data loaded.")
            if hasattr(self, "infob"):
                self.infob.setText("")
            if hasattr(self, "infoc"):
                self.infoc.setText("")
            self.pconnected = 0
            self.nedges = 0
            self.graph = None
            if hasattr(self, "sendSignals"):
                self.sendSignals()
            return

        nEdgesEstimate = 2 * sum(y for x, y in zip(self.histogram.xData, self.histogram.yData)
                                 if self.spinLowerThreshold <= x <= self.spinUpperThreshold)

        if nEdgesEstimate > 200000:
            self.graph = None
            nedges = 0
            n = 0
            self.error('Estimated number of edges is too high (%d).' % nEdgesEstimate)
        else:
            graph = network.Graph()
            graph.add_nodes_from(range(self.matrix.shape[0]))
            matrix = self.matrix

            if hasattr(self.matrix, "items") and self.matrix.items is not None:
                if isinstance(self.matrix.items, Orange.data.Table):
                    graph.set_items(self.matrix.items)
                else:
                    data = [[str(x)] for x in self.matrix.items]
                    items = Orange.data.Table(Orange.data.Domain([], metas=[Orange.data.StringVariable('label')]), data)
                    graph.set_items(items)

            # set the threshold
            # set edges where distance is lower than threshold
            self.warning(0)
            if self.kNN >= self.matrix.shape[0]:
                self.warning(0, "kNN larger then supplied distance matrix dimension. Using k = %i" % (self.matrix.shape[0] - 1))
            #nedges = graph.fromDistanceMatrix(self.matrix, self.spinLowerThreshold, self.spinUpperThreshold, min(self.kNN, self.matrix.shape[0] - 1), self.andor)

            def edges_from_distance_matrix(matrix, lower, upper, knn):
                rows, cols = matrix.shape
                if knn:
                    for i in range(rows):
                        for j in np.argsort(matrix[i])[:knn]:
                            yield i, j, matrix[i, j]
                else:
                    for i in range(rows):
                        for j in range(i + 1, cols):
                            if lower <= matrix[i, j] <= upper:
                                yield i, j, matrix[i, j]

            edge_list = edges_from_distance_matrix(
                self.matrix, self.spinLowerThreshold, self.spinUpperThreshold,
                min(self.kNN, self.matrix.shape[0] - 1))
            if self.dstWeight == 1:
                graph.add_edges_from(((u, v, {'weight':1 - d}) for u, v, d in edge_list))
            else:
                graph.add_edges_from(((u, v, {'weight':d}) for u, v, d in edge_list))

            matrix = None
            self.graph = None
            # exclude unconnected
            if str(self.netOption) == '1':
                components = [x for x in network.nx.algorithms.components.connected_components(graph) if len(x) >= self.excludeLimit]
                if len(components) > 0:
                    include = list(chain.from_iterable(components))
                    if len(include) > 1:
                        self.graph = graph.subgraph(include)
                        matrix = self.matrix.submatrix(include)
            # largest connected component only
            elif str(self.netOption) == '2':
                component = next(network.nx.algorithms.components.connected_components(graph))
                if len(component) > 1:
                    self.graph = graph.subgraph(component)
                    matrix = self.matrix.submatrix(component)
            # connected component with vertex by label
            # elif str(self.netOption) == '3':
            #     self.attributeCombo.box.setEnabled(True)
            #     self.graph = None
            #     matrix = None
            #     if self.attributeCombo.currentText() != '' and self.label != '':
            #         components = network.nx.algorithms.components.connected_components(graph)

            #         txt = self.label.lower()
            #         #print 'txt:',txt
            #         nodes = [i for i, values in enumerate(self.matrix.items) if txt in str(values[str(self.attributeCombo.currentText())]).lower()]
            #         #print "nodes:",nodes
            #         if len(nodes) > 0:
            #             vertices = []
            #             for component in components:
            #                 for node in nodes:
            #                     if node in component:
            #                         if len(component) > 0:
            #                             vertices.extend(component)

            #             if len(vertices) > 0:
            #                 #print "n vertices:", len(vertices), "n set vertices:", len(set(vertices))
            #                 vertices = list(set(vertices))
            #                 self.graph = graph.subgraph(include)
            #                 matrix = self.matrix.getitems(vertices)
            else:
                self.graph = graph

        if matrix != None:
            setattr(matrix, "items", self.graph.items())
        self.graph_matrix = matrix

        if self.graph is None:
            self.pconnected = 0
            self.nedges = 0
        else:
            self.pconnected = self.graph.number_of_nodes()
            self.nedges = self.graph.number_of_edges()
        if hasattr(self, "infoa"):
            self.infoa.setText("Data items on input: %d" % self.matrix.shape[0])
        if hasattr(self, "infob"):
            self.infob.setText("Network nodes: %d (%3.1f%%)" % (self.pconnected,
                self.pconnected / float(self.matrix.shape[0]) * 100))
        if hasattr(self, "infoc"):
            self.infoc.setText("Network edges: %d (%.2f edges/node)" % (
                self.nedges, self.nedges / float(self.pconnected)
                if self.pconnected else 0))

        if hasattr(self, "sendSignals"):
            self.sendSignals()

        self.histogram.setRegion(self.spinLowerThreshold, self.spinUpperThreshold)


class Histogram(pg.PlotWidget):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, setAspectLocked=True, **kwargs)
        self.curve = self.plot([0, 1], [0], pen=pg.mkPen('b', width=2), stepMode=True)
        self.region = pg.LinearRegionItem([0, 0], brush=pg.mkBrush('#02f1'), movable=True)
        self.region.sigRegionChanged.connect(self._update_region)
        self.addItem(self.region)
        self.fillCurve = self.plotItem.plot([0, 1], [0],
            fillLevel=0, pen=pg.mkPen('b', width=2), brush='#02f3', stepMode=True)
        self.plotItem.vb.setMouseEnabled(x=False, y=False)

    def _update_region(self, region):
        rlow, rhigh = self.getRegion()
        low = max(0, np.searchsorted(self.xData, rlow, side='right') - 1)
        high = np.searchsorted(self.xData, rhigh, side='right')
        if high - low > 0:
            xData = self.xData[low:high + 1].copy()
            xData[0] = rlow  # set visible boundaries to match region lines
            xData[-1] = rhigh
            self.fillCurve.setData(xData, self.yData[low:high])

    def setBoundary(self, low, high):
        self.region.setBounds((low, high))

    def boundary(self):
        return self.xData[[0, -1]]

    def setRegion(self, low, high):
        self.region.setRegion((low, high))

    def getRegion(self):
        return self.region.getRegion()

    def setValues(self, values):
        if not len(values):
            self.curve.setData([0, 1], [0])
            self.setBoundary(0, 0)
            return
        nbins = min(np.sqrt(len(values)), len(values))
        freq, edges = np.histogram(values, bins=nbins)
        self.curve.setData(edges, freq)
        self.setBoundary(edges[0], edges[-1])

    @property
    def xData(self):
        return self.curve.xData

    @property
    def yData(self):
        return self.curve.yData
