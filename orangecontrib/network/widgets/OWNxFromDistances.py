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
from Orange.widgets import gui, widget, settings
import orangecontrib.network as network

import pyqtgraph as pg


class NodeSelection:
    ALL_NODES,   \
    COMPONENTS,  \
    LARGEST_COMP = range(3)

class EdgeWeights:
    PROPORTIONAL, \
    INVERSE = range(2)


class OWNxFromDistances(widget.OWWidget):
    name = "Network from Distances"
    description = ('Constructs Graph object by connecting nodes from '
                   'data table where distance between them is between '
                   'given threshold.')
    icon = "icons/NetworkFromDistances.svg"
    priority = 6440

    inputs = [("Distances", Orange.misc.DistMatrix, "setMatrix")]
    outputs = [("Network", network.Graph),
               ("Data", Orange.data.Table),
               ("Distances", Orange.misc.DistMatrix)]

    resizing_enabled = False

    # TODO: make settings input-dependent
    percentil = settings.Setting(1)
    include_knn = settings.Setting(False)
    kNN = settings.Setting(2)
    node_selection = settings.Setting(0)
    edge_weights = settings.Setting(0)
    excludeLimit = settings.Setting(2)

    def __init__(self):
        super().__init__()

        self.spinUpperThreshold = 0

        self.matrix = None
        self.graph = None
        self.graph_matrix = None

        self.histogram = Histogram(self)
        self.mainArea.layout().addWidget(self.histogram)
        self.mainArea.setMinimumWidth(500)
        self.mainArea.setMinimumHeight(300)
        self.addHistogramControls()

        # info
        boxInfo = gui.widgetBox(self.controlArea, box="Info")
        self.infoa = gui.widgetLabel(boxInfo, "No data loaded.")
        self.infob = gui.widgetLabel(boxInfo, '')
        self.infoc = gui.widgetLabel(boxInfo, '')

        gui.rubber(self.controlArea)

        self.resize(700, 100)

    def addHistogramControls(self):
        boxGeneral = gui.widgetBox(self.controlArea, box="Edges")
        ribg = gui.widgetBox(boxGeneral, None, orientation="horizontal", addSpace=False)
        self.spin_high = gui.doubleSpin(boxGeneral, self, 'spinUpperThreshold',
                                        0, float('inf'), 0.001, decimals=3,
                                        label='Distance threshold',
                                        callback=self.changeUpperSpin,
                                        keyboardTracking=False,
                                        controlWidth=60)
        self.histogram.region.sigRegionChangeFinished.connect(self.spinboxFromHistogramRegion)

        ribg = gui.widgetBox(boxGeneral, None, orientation="horizontal", addSpace=False)

        gui.doubleSpin(boxGeneral, self, "percentil", 0, 100, 0.1,
                      label="Percentile", orientation='horizontal',
                      callback=self.setPercentil,
                      callbackOnReturn=1, controlWidth=60)

        hbox = gui.widgetBox(boxGeneral, orientation='horizontal')
        knn_cb = gui.checkBox(hbox, self, 'include_knn',
                              label='Include also closest neighbors',
                              callback=self.generateGraph)
        knn = gui.spin(hbox, self, "kNN", 1, 1000, 1,
                       orientation='horizontal',
                       callback=self.generateGraph, callbackOnReturn=1, controlWidth=60)
        knn_cb.disables = [knn]
        knn_cb.makeConsistent()

        ribg.layout().addStretch(1)
        # Options
        ribg = gui.radioButtonsInBox(self.controlArea, self, "node_selection",
                                     box="Node selection",
                                     callback=self.generateGraph)
        gui.appendRadioButton(ribg, "Keep all nodes")
        hb = gui.widgetBox(ribg, None, orientation="horizontal", addSpace=False)
        gui.appendRadioButton(ribg, "Components with at least nodes", insertInto=hb)
        gui.spin(hb, self, "excludeLimit", 2, 100, 1,
                 callback=(lambda h=True: self.generateGraph(h)), controlWidth=60)
        gui.appendRadioButton(ribg, "Largest connected component")
        self.attribute = None

        ribg = gui.radioButtonsInBox(self.controlArea, self, "edge_weights",
                                     box="Edge weights",
                                     callback=self.generateGraph)
        hb = gui.widgetBox(ribg, None, orientation="horizontal", addSpace=False)
        gui.appendRadioButton(ribg, "Proportional to distance", insertInto=hb)
        gui.appendRadioButton(ribg, "Inverted distance", insertInto=hb)

    def setPercentil(self):
        if self.matrix is None or self.percentil <= 0:
            return
        # flatten matrix, sort values and remove identities (self.matrix[i][i])
        ind = int(len(self.matrix_values) * self.percentil / 100)
        self.spinUpperThreshold = self.matrix_values[ind]
        self.generateGraph()

    def setMatrix(self, data):
        self.matrix = data
        if data is None:
            self.histogram.setValues([])
            self.generateGraph()
            return

        if self.matrix.row_items is None:
            self.matrix.row_items = list(range(self.matrix.shape[0]))

        # draw histogram
        self.matrix_values = values = sorted(self.matrix.flat)
        self.histogram.setValues(values)

        # Magnitude of the spinbox's step is data-dependent
        low, upp = values[0], values[-1]
        step = (upp - low) / 20
        self.spin_high.setSingleStep(step)

        self.spinUpperThreshold = low - (0.03 * (upp - low))

        self.setPercentil()
        self.generateGraph()

    def changeUpperSpin(self):
        if self.matrix is None: return
        self.spinUpperThreshold = np.clip(self.spinUpperThreshold, *self.histogram.boundary())
        self.percentil = 100 * np.searchsorted(self.matrix_values, self.spinUpperThreshold) / len(self.matrix_values)
        self.generateGraph()

    def spinboxFromHistogramRegion(self):
        _, self.spinUpperThreshold = self.histogram.getRegion()
        self.changeUpperSpin()

    def generateGraph(self, N_changed=False):
        self.error()
        matrix = None
        self.warning('')

        if N_changed:
            self.node_selection = NodeSelection.COMPONENTS

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
            self.sendSignals()
            return

        nEdgesEstimate = 2 * sum(y for x, y in zip(self.histogram.xData, self.histogram.yData)
                                 if x <= self.spinUpperThreshold)

        if nEdgesEstimate > 200000:
            self.graph = None
            nedges = 0
            n = 0
            self.error('Estimated number of edges is too high (%d).' % nEdgesEstimate)
        else:
            graph = network.Graph()
            graph.add_nodes_from(range(self.matrix.shape[0]))
            matrix = self.matrix

            if matrix is not None and matrix.row_items is not None:
                if isinstance(self.matrix.row_items, Orange.data.Table):
                    graph.set_items(self.matrix.row_items)
                else:
                    data = [[str(x)] for x in self.matrix.row_items]
                    items = Orange.data.Table(Orange.data.Domain([], metas=[Orange.data.StringVariable('label')]), data)
                    graph.set_items(items)

            # set the threshold
            # set edges where distance is lower than threshold
            self.warning(0)
            if self.kNN >= self.matrix.shape[0]:
                self.warning(0, "kNN larger then supplied distance matrix dimension. Using k = %i" % (self.matrix.shape[0] - 1))

            def edges_from_distance_matrix(matrix, upper, knn):
                rows, cols = matrix.shape
                for i in range(rows):
                    for j in range(i + 1, cols):
                        if matrix[i, j] <= upper:
                            yield i, j, matrix[i, j]
                    if not knn: continue
                    for j in np.argsort(matrix[i])[:knn]:
                        yield i, j, matrix[i, j]

            edge_list = edges_from_distance_matrix(
                self.matrix, self.spinUpperThreshold,
                min(self.kNN, self.matrix.shape[0] - 1) if self.include_knn else 0)
            if self.edge_weights == EdgeWeights.INVERSE:
                edge_list = list(edge_list)
                max_weight = max(d for u, v, d in edge_list)
                graph.add_edges_from((u, v, {'weight': max_weight - d})
                                     for u, v, d in edge_list)
            else:
                graph.add_edges_from((u, v, {'weight': d})
                                     for u, v, d in edge_list)
            matrix = None
            self.graph = None
            component = []
            # exclude unconnected
            if self.node_selection == NodeSelection.COMPONENTS:
                component = list(chain.from_iterable(x for x in network.nx.connected_components(graph)
                                                     if len(x) >= self.excludeLimit))
            # largest connected component only
            elif self.node_selection == NodeSelection.LARGEST_COMP:
                component = max(network.nx.connected_components(graph), key=len)
            else:
                self.graph = graph
            if len(component) > 1:
                if len(component) == graph.number_of_nodes():
                    self.graph = graph
                    matrix = self.matrix
                else:
                    self.graph = graph.subgraph(component)
                    matrix = self.matrix.submatrix(sorted(component))

        if matrix is not None:
            matrix.row_items = self.graph.items()
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

        self.warning(303)
        if self.pconnected > 1000 or self.nedges > 2000:
            self.warning(303, 'Large number of nodes/edges; performance will be hindered.')

        self.sendSignals()
        self.histogram.setRegion(0, self.spinUpperThreshold)

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Edge thresholds", "%.5f - %.5f" % \
                              (0, \
                               self.spinUpperThreshold)),
                             ("Selected vertices", ["All", \
                                "Without isolated vertices",
                                "Largest component",
                                "Connected with vertex"][self.node_selection]),
                             ("Weight", ["Distance", "1 - Distance"][self.edge_weights])])
        self.reportSection("Histogram")
        self.reportImage(self.histogram.saveToFileDirect, QSize(400,300))
        self.reportSettings("Output graph",
                            [("Vertices", self.matrix.dim),
                             ("Edges", self.nedges),
                             ("Connected vertices", "%i (%.1f%%)" % \
                              (self.pconnected, self.pconnected / \
                               max(1, float(self.matrix.dim))*100))])

    def sendSignals(self):
        self.send("Network", self.graph)
        self.send("Distances", self.graph_matrix)
        if self.graph == None:
            self.send("Data", None)
        else:
            self.send("Data", self.graph.items())


pg_InfiniteLine = pg.InfiniteLine

class InfiniteLine(pg_InfiniteLine):
    def paint(self, p, *args):
        # From orange3-bioinformatics:OWFeatureSelection.py, thanks to @ales-erjavec
        brect = self.boundingRect()
        c = brect.center()
        line = QLineF(brect.left(), c.y(), brect.right(), c.y())
        t = p.transform()
        line = t.map(line)
        p.save()
        p.resetTransform()
        p.setPen(self.currentPen)
        p.drawLine(line)
        p.restore()

# Patched so that the Histogram's LinearRegionItem works on MacOS
pg.InfiniteLine = InfiniteLine
pg.graphicsItems.LinearRegionItem.InfiniteLine = InfiniteLine


class Histogram(pg.PlotWidget):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, setAspectLocked=True, **kwargs)
        self.curve = self.plot([0, 1], [0], pen=pg.mkPen('b', width=2), stepMode=True)
        self.region = pg.LinearRegionItem([0, 0], brush=pg.mkBrush('#02f1'), movable=True)
        self.region.sigRegionChanged.connect(self._update_region)
        # Selected region is only open-ended on the the upper side
        self.region.hoverEvent = self.region.mouseDragEvent = lambda *args: None
        self.region.lines[0].setVisible(False)
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
        low, high = np.clip([low, high], *self.boundary())
        self.region.setRegion((low, high))

    def getRegion(self):
        return self.region.getRegion()

    def setValues(self, values):
        self.fillCurve.setData([0,1], [0])
        if not len(values):
            self.curve.setData([0, 1], [0])
            self.setBoundary(0, 0)
            return
        nbins = int(min(np.sqrt(len(values)), 100))
        freq, edges = np.histogram(values, bins=nbins)
        self.curve.setData(edges, freq)
        self.setBoundary(edges[0], edges[-1])
        self.autoRange()

    @property
    def xData(self):
        return self.curve.xData

    @property
    def yData(self):
        return self.curve.yData


if __name__ == "__main__":
    appl = QApplication([])
    ow = OWNxFromDistances()
    ow.show()
    appl.exec_()
