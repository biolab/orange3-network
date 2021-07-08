import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

import pyqtgraph as pg

from AnyQt.QtCore import QLineF, QSize, Qt
from AnyQt.QtWidgets import QGridLayout, QLabel

from Orange.data import Domain, StringVariable, Table
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview
from orangecontrib.network.network import Network


class NodeSelection:
    ALL_NODES,   \
    COMPONENTS,  \
    LARGEST_COMP = range(3)

class EdgeWeights:
    PROPORTIONAL, \
    INVERSE = range(2)

PERCENTIL_STEP = 0.1

class OWNxFromDistances(widget.OWWidget):
    name = "Network From Distances"
    description = ('Constructs Graph object by connecting nodes from '
                   'data table where distance between them is between '
                   'given threshold.')
    icon = "icons/NetworkFromDistances.svg"
    priority = 6440

    class Inputs:
        distances = Input("Distances", DistMatrix)

    class Outputs:
        network = Output("Network", Network)
        data = Output("Data", Table)
        distances = Output("Distances", DistMatrix)

    resizing_enabled = False

    # TODO: make settings input-dependent
    percentil = settings.Setting(1)
    include_knn = settings.Setting(False)
    kNN = settings.Setting(2)
    node_selection = settings.Setting(0)
    edge_weights = settings.Setting(0)
    excludeLimit = settings.Setting(2)

    class Warning(widget.OWWidget.Warning):
        kNN_too_large = \
            Msg('kNN is larger than supplied distance matrix dimension. '
                'Using k = {}')
        large_number_of_nodes = \
            Msg('Large number of nodes/edges; performance will be hindered')
        invalid_number_of_items = \
            Msg('Number of data items does not match the nunmber of nodes')

    class Error(widget.OWWidget.Error):
        number_of_edges = Msg('Estimated number of edges is too high ({})')

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
        # gui.spin and gui.doubleSpin cannot not be inserted somewhere, thus
        # we insert them into controlArea and then move to grid
        grid = QGridLayout()
        gui.widgetBox(self.controlArea, box="Edges", orientation=grid)
        self.spin_high = gui.doubleSpin(self.controlArea, self, 'spinUpperThreshold',
                                        0, float('inf'), 0.001, decimals=3,
                                        callback=self.changeUpperSpin,
                                        keyboardTracking=False,
                                        controlWidth=60,
                                        addToLayout=False)
        grid.addWidget(QLabel("Distance threshold"), 0, 0)
        grid.addWidget(self.spin_high, 0, 1)

        self.histogram.region.sigRegionChangeFinished.connect(self.spinboxFromHistogramRegion)

        spin = gui.doubleSpin(self.controlArea, self, "percentil", 0, 100, PERCENTIL_STEP,
                      label="Percentile", orientation=Qt.Horizontal,
                      callback=self.setPercentil,
                      callbackOnReturn=1, controlWidth=60)
        grid.addWidget(QLabel("Percentile"), 1, 0)
        grid.addWidget(spin, 1, 1)

        knn_cb = gui.checkBox(self.controlArea, self, 'include_knn',
                              label='Include closest neighbors',
                              callback=self.generateGraph)
        knn = gui.spin(self.controlArea, self, "kNN", 1, 1000, 1,
                       orientation=Qt.Horizontal,
                       callback=self.generateGraph, callbackOnReturn=1, controlWidth=60)
        grid.addWidget(knn_cb, 2, 0)
        grid.addWidget(knn, 2, 1)

        knn_cb.disables = [knn]
        knn_cb.makeConsistent()

        # Options
        ribg = gui.radioButtonsInBox(self.controlArea, self, "node_selection",
                                     box="Node selection",
                                     callback=self.generateGraph)
        grid = QGridLayout()
        ribg.layout().addLayout(grid)
        grid.addWidget(
            gui.appendRadioButton(ribg, "Keep all nodes", addToLayout=False),
            0, 0
        )

        exclude_limit = gui.spin(
            ribg, self, "excludeLimit", 2, 100, 1,
            callback=(lambda h=True: self.generateGraph(h)),
            controlWidth=60
        )
        grid.addWidget(
            gui.appendRadioButton(ribg, "Components with at least nodes",
                                  addToLayout=False), 1, 0)
        grid.addWidget(exclude_limit, 1, 1)

        grid.addWidget(
            gui.appendRadioButton(ribg, "Largest connected component",
                                  addToLayout=False), 2, 0)

        ribg = gui.radioButtonsInBox(self.controlArea, self, "edge_weights",
                                     box="Edge weights",
                                     callback=self.generateGraph)
        hb = gui.widgetBox(ribg, None, addSpace=False)
        gui.appendRadioButton(ribg, "Proportional to distance", insertInto=hb)
        gui.appendRadioButton(ribg, "Inverted distance", insertInto=hb)

    def setPercentil(self):
        # Correct 0th and 100th percentile to min and max
        if self.percentil < (0 + PERCENTIL_STEP):
            self.percentil = 1 / self.matrix.shape[0] * 100.0
        elif self.percentil > (100 - PERCENTIL_STEP):
            self.percentil = (self.matrix.shape[0] - 1) / self.matrix.shape[0] * 100.0

        if self.matrix is None:
            return
        if len(self.matrix_values) > 0:
            # flatten matrix, sort values and remove identities (self.matrix[i][i])
            ind = int(len(self.matrix_values) * self.percentil / 100)
            self.spinUpperThreshold = self.matrix_values[ind]
        self.generateGraph()

    @Inputs.distances
    def set_matrix(self, data):
        if data is not None and not data.size:
            data = None
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
        if len(values) == 0:
            low, upp = 0, 0
        else:
            low, upp = values[0], values[-1]
        step = (upp - low) / 20
        self.spin_high.setSingleStep(step)

        self.spinUpperThreshold = max(0, low - (0.03 * (upp - low)))

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
        self.Error.clear()
        self.Warning.clear()
        matrix = None

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
            graph = None
            self.Error.number_of_edges(nEdgesEstimate)
        else:
            items = None
            matrix = self.matrix
            if matrix is not None and matrix.row_items is not None:
                row_items = self.matrix.row_items
                if isinstance(row_items, Table):
                    if self.matrix.axis == 1:
                        items = row_items
                    else:
                        items = [[v.name] for v in row_items.domain.attributes]
                else:
                    items = [[str(x)] for x in self.matrix.row_items]
            if len(items) != self.matrix.shape[0]:
                self.Warning.invalid_number_of_items()
                items = None
            if items is None:
                items = list(range(self.matrix.shape[0]))
            if not isinstance(items, Table):
                items = Table.from_list(Domain([], metas=[StringVariable('label')]), items)

            # set the threshold
            # set edges where distance is lower than threshold
            self.Warning.kNN_too_large.clear()
            if self.kNN >= self.matrix.shape[0]:
                self.Warning.kNN_too_large(self.matrix.shape[0] - 1)

            mask = np.array(self.matrix <= self.spinUpperThreshold)
            if self.include_knn:
                mask |= mask.argsort() < self.kNN
            np.fill_diagonal(mask, 0)
            weights = matrix[mask]
            shape = (len(items), len(items))
            if weights.size:
                if self.edge_weights == EdgeWeights.INVERSE:
                    weights = np.max(weights) - weights
                edges = sp.csr_matrix((weights, mask.nonzero()), shape=shape)
            else:
                edges = sp.csr_matrix(shape)
            graph = Network(items, edges)

            self.graph = None
            # exclude unconnected
            if self.node_selection != NodeSelection.ALL_NODES:
                n_components, components = csgraph.connected_components(edges)
                counts = np.bincount(components)
                if self.node_selection == NodeSelection.COMPONENTS:
                    ind = np.flatnonzero(counts >= self.excludeLimit)
                    mask = np.in1d(components, ind)
                else:
                    mask = components == np.argmax(counts)
                graph = graph.subgraph(mask)

        self.graph = graph
        if graph is None:
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

        self.Warning.large_number_of_nodes.clear()
        if self.pconnected > 1000 or self.nedges > 2000:
            self.Warning.large_number_of_nodes()

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
        self.Outputs.network.send(self.graph)
        self.Outputs.distances.send(self.graph_matrix)
        if self.graph is None:
            self.Outputs.data.send(None)
        else:
            self.Outputs.data.send(self.graph.nodes)


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
        super().__init__(parent, setAspectLocked=True, background="w", **kwargs)
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
    WidgetPreview(OWNxFromDistances).run(set_matrix=(Euclidean(Table("iris"))))
