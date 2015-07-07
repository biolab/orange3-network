#
# OWHist.py
#
# the base for network histograms

import math
import numpy as np

import Orange
from Orange.widgets import gui, widget
import orangecontrib.network as network

from functools import reduce





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
        gui.doubleSpin(ribg, self, "spinLowerThreshold",
                         0.0, float("inf"), 0.0001, decimals=4,
                         callback=self.changeLowerSpin,
                         keyboardTracking=False)
        ribg.layout().addWidget(QLabel("to", self), 1, Qt.AlignCenter)
        gui.doubleSpin(ribg, self, "spinUpperThreshold",
                         0.0, float("inf"), 0.0001, decimals=4,
                         callback=self.changeUpperSpin,
                         keyboardTracking=False)
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
        gui.appendRadioButton(ribg, self, "netOption", "Keep all nodes", callback=self.generateGraph)
        hb = gui.widgetBox(ribg, None, orientation="horizontal", addSpace=False)
        gui.appendRadioButton(ribg, self, "netOption", "Components with at least nodes", insertInto=hb, callback=self.generateGraph)
        gui.spin(hb, self, "excludeLimit", 2, 100, 1, callback=(lambda h=True: self.generateGraph(h)), controlWidth=60)
        gui.appendRadioButton(ribg, self, "netOption", "Largest connected component", callback=self.generateGraph)
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
        gui.appendRadioButton(ribg, self, "dstWeight", "Proportional to distance", insertInto=hb, callback=self.generateGraph)
        gui.appendRadioButton(ribg, self, "dstWeight", "Inverted distance", insertInto=hb, callback=self.generateGraph)



    def setPercentil(self):
        if self.matrix is None or self.percentil <= 0:
            return

        self.spinLowerThreshold = self.histogram.minValue
        # flatten matrix, sort values and remove identities (self.matrix[i][i])
        vals = sorted(sum(self.matrix, ()))[self.matrix.dim:]
        ind = int(len(vals) * self.percentil / 100)
        self.spinUpperThreshold = vals[ind]
        self.generateGraph()

    def enableAttributeSelection(self):
        #self.attributeCombo.box.setEnabled(True)
        pass

    def setSearchStringTimer(self):
        # self.searchStringTimer.stop()
        # self.searchStringTimer.start(750)
        pass

    def setMatrix(self, data):
        if data == None:
            self.matrix = None
            self.histogram.setValues([])
            #self.attributeCombo.clear()
            self.generateGraph()
            return

        if not hasattr(data, "items") or data.items is None:
            setattr(data, "items", [i for i in range(data.dim)])

        self.matrix = data
        # draw histogram
        values = data.getValues()
        #print "values:",values
        self.histogram.setValues(values)

        low = min(values)
        upp = max(values)

        self.spinLowerThreshold = self.spinUpperThreshold = low - (0.03 * (upp - low))

        # self.attributeCombo.clear()
        vars = []

        if hasattr(self.matrix, "items"):

            if isinstance(self.matrix.items, orange.ExampleTable):
                vars = list(self.matrix.items.domain.variables)

                metas = self.matrix.items.domain.getmetas(0)
                for i, var in metas.items():
                    vars.append(var)

        self.icons = self.createAttributeIconDict()

        # for var in vars:
        #     try:
        #         if var.varType != 7: # if not Orange.feature.Python
        #             self.attributeCombo.addItem(self.icons[var.varType], unicode(var.name))
        #     except:
        #         print "Error adding", var, "to the attribute combo."

        self.setPercentil()
        self.generateGraph()

    def changeLowerSpin(self):
        self.percentil = 0

        if self.spinLowerThreshold < self.histogram.minValue:
            self.spinLowerThreshold = self.histogram.minValue
        elif self.spinLowerThreshold > self.histogram.maxValue:
            self.spinLowerThreshold = self.histogram.maxValue

        if self.spinLowerThreshold >= self.spinUpperThreshold:
            self.spinUpperThreshold = self.spinLowerThreshold

        self.generateGraph()

    def changeUpperSpin(self):
        self.percentil = 0

        if self.spinUpperThreshold < self.histogram.minValue:
            self.spinUpperThreshold = self.histogram.minValue
        elif self.spinUpperThreshold > self.histogram.maxValue:
            self.spinUpperThreshold = self.histogram.maxValue

        if self.spinUpperThreshold <= self.spinLowerThreshold:
            self.spinLowerThreshold = self.spinUpperThreshold

        self.generateGraph()

    def generateGraph(self, N_changed=False):
        # self.searchStringTimer.stop()
        # self.attributeCombo.box.setEnabled(False)
        self.error()
        matrix = None
        self.warning('')

        if N_changed:
            self.netOption = 1

        if self.matrix == None:
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

        #print len(self.histogram.yData), len(self.histogram.xData)
        nEdgesEstimate = 2 * sum([self.histogram.yData[i] for i, e in enumerate(self.histogram.xData) if self.spinLowerThreshold <= e <= self.spinUpperThreshold])

        if nEdgesEstimate > 200000:
            self.graph = None
            nedges = 0
            n = 0
            self.error('Estimated number of edges is too high (%d).' % nEdgesEstimate)
        else:
            graph = network.Graph()
            graph.add_nodes_from(range(self.matrix.dim))
            matrix = self.matrix

            if hasattr(self.matrix, "items") and self.matrix.items is not None:
                if type(self.matrix.items) == Orange.data.Table:
                    graph.set_items(self.matrix.items)
                else:
                    data = [[str(x)] for x in self.matrix.items]
                    items = Orange.data.Table(Orange.data.Domain(Orange.data.StringVariable('label'), 0), data)
                    graph.set_items(items)

            # set the threshold
            # set edges where distance is lower than threshold
            self.warning(0)
            if self.kNN >= self.matrix.dim:
                self.warning(0, "kNN larger then supplied distance matrix dimension. Using k = %i" % (self.matrix.dim - 1))
            #nedges = graph.fromDistanceMatrix(self.matrix, self.spinLowerThreshold, self.spinUpperThreshold, min(self.kNN, self.matrix.dim - 1), self.andor)
            edge_list = network.GraphLayout().edges_from_distance_matrix(self.matrix, self.spinLowerThreshold, self.spinUpperThreshold, min(self.kNN, self.matrix.dim - 1))
            if self.dstWeight == 1:
                graph.add_edges_from(((u, v, {'weight':1 - d}) for u, v, d in edge_list))
            else:
                graph.add_edges_from(((u, v, {'weight':d}) for u, v, d in edge_list))

            # exclude unconnected
            if str(self.netOption) == '1':
                components = [x for x in network.nx.algorithms.components.connected_components(graph) if len(x) >= self.excludeLimit]
                if len(components) > 0:
                    include = reduce(lambda x, y: x + y, components)
                    if len(include) > 1:
                        self.graph = graph.subgraph(include)
                        matrix = self.matrix.getitems(include)
                    else:
                        self.graph = None
                        matrix = None
                else:
                    self.graph = None
                    matrix = None
            # largest connected component only
            elif str(self.netOption) == '2':
                component = network.nx.algorithms.components.connected_components(graph)[0]
                if len(component) > 1:
                    self.graph = graph.subgraph(component)
                    matrix = self.matrix.getitems(component)
                else:
                    self.graph = None
                    matrix = None
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
            matrix.setattr("items", self.graph.items())
        self.graph_matrix = matrix

        if self.graph is None:
            self.pconnected = 0
            self.nedges = 0
        else:
            self.pconnected = self.graph.number_of_nodes()
            self.nedges = self.graph.number_of_edges()
        if hasattr(self, "infoa"):
            self.infoa.setText("Data items on input: %d" % self.matrix.dim)
        if hasattr(self, "infob"):
            self.infob.setText("Network nodes: %d (%3.1f%%)" % (self.pconnected,
                self.pconnected / float(self.matrix.dim) * 100))
        if hasattr(self, "infoc"):
            self.infoc.setText("Network edges: %d (%.2f edges/node)" % (
                self.nedges, self.nedges / float(self.pconnected)
                if self.pconnected else 0))

        #print 'self.graph:',self.graph+
        if hasattr(self, "sendSignals"):
            self.sendSignals()

        self.histogram.setBoundary(self.spinLowerThreshold, self.spinUpperThreshold)





# All below shamefully copied from orangecontrib.bio.widgets3.OWFeatureSelection
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot


class InfiniteLine(pg.InfiniteLine):
    def paint(self, painter, option, widget=None):
        brect = self.boundingRect()
        c = brect.center()
        line = QtCore.QLineF(brect.left(), c.y(), brect.right(), c.y())
        t = painter.transform()
        line = t.map(line)
        painter.save()
        painter.resetTransform()
        painter.setPen(self.currentPen)
        painter.drawLine(line)
        painter.restore()


class Histogram(pg.PlotWidget):
    """
    A histogram plot with interactive 'tail' selection
    """
    #: Emitted when the selection boundary has changed
    selectionChanged = Signal()
    #: Emitted when the selection boundary has been edited by the user
    #: (by dragging the boundary lines)
    selectionEdited = Signal()

    #: Selection mode
    NoSelection, Low, High, TwoSided, Middle = 0, 1, 2, 3, 4


    """Adendum: compliance with OWNxFromDistances"""
    def setValues(self, values):
        ...

    def


    """End Adendum"""

    def __init__(self, parent=None, **kwargs):
        pg.PlotWidget.__init__(self, parent, **kwargs)

        self.getAxis("bottom").setLabel("Score")
        self.getAxis("left").setLabel("Counts")

        self.__data = None
        self.__histcurve = None

        self.__mode = Histogram.NoSelection
        self.__min = 0
        self.__max = 0

        def makeline(pos):
            pen = QtGui.QPen(Qt.darkGray, 1)
            pen.setCosmetic(True)
            line = InfiniteLine(angle=90, pos=pos, pen=pen, movable=True)
            line.setCursor(Qt.SizeHorCursor)
            return line

        self.__cuthigh = makeline(self.__max)
        self.__cuthigh.sigPositionChanged.connect(self.__on_cuthigh_changed)
        self.__cuthigh.sigPositionChangeFinished.connect(self.selectionEdited)
        self.__cutlow = makeline(self.__min)
        self.__cutlow.sigPositionChanged.connect(self.__on_cutlow_changed)
        self.__cutlow.sigPositionChangeFinished.connect(self.selectionEdited)

        brush = pg.mkBrush((200, 200, 200, 180))
        self.__taillow = pg.PlotCurveItem(
            fillLevel=0, brush=brush, pen=QtGui.QPen(Qt.NoPen))
        self.__taillow.setVisible(False)

        self.__tailhigh = pg.PlotCurveItem(
            fillLevel=0, brush=brush, pen=QtGui.QPen(Qt.NoPen))
        self.__tailhigh.setVisible(False)

    def setData(self, hist, bins=None):
        """
        Set the histogram data
        """
        if bins is None:
            bins = np.arange(len(hist))

        self.__data = (hist, bins)
        if self.__histcurve is None:
            self.__histcurve = pg.PlotCurveItem(
                x=bins, y=hist, stepMode=True
            )
        else:
            self.__histcurve.setData(x=bins, y=hist, stepMode=True)

        self.__update()

    def setHistogramCurve(self, curveitem):
        """
        Set the histogram plot curve.
        """
        if self.__histcurve is curveitem:
            return

        if self.__histcurve is not None:
            self.removeItem(self.__histcurve)
            self.__histcurve = None
            self.__data = None

        if curveitem is not None:
            if not curveitem.opts["stepMode"]:
                raise ValueError("The curve must have `stepMode == True`")
            self.addItem(curveitem)
            self.__histcurve = curveitem
            self.__data = (curveitem.yData, curveitem.xData)

        self.__update()

    def histogramCurve(self):
        """
        Return the histogram plot curve.
        """
        return self.__histcurve

    def setSelectionMode(self, mode):
        """
        Set selection mode
        """
        if self.__mode != mode:
            self.__mode = mode
            self.__update_cutlines()
            self.__update_tails()

    def setLower(self, value):
        """
        Set the lower boundary value.
        """
        if self.__min != value:
            self.__min = value
            self.__update_cutlines()
            self.__update_tails()
            self.selectionChanged.emit()

    def setUpper(self, value):
        """
        Set the upper boundary value.
        """
        if self.__max != value:
            self.__max = value
            self.__update_cutlines()
            self.__update_tails()
            self.selectionChanged.emit()

    def setBoundary(self, lower, upper):
        """
        Set lower and upper boundary value.
        """
        changed = False
        if self.__min != lower:
            self.__min = lower
            changed = True

        if self.__max != upper:
            self.__max = upper
            changed = True

        if changed:
            self.__update_cutlines()
            self.__update_tails()
            self.selectionChanged.emit()

    def boundary(self):
        """
        Return the lower and upper boundary values.
        """
        return (self.__min, self.__max)

    def clear(self):
        """
        Clear the plot.
        """
        self.__data = None
        self.__histcurve = None
        super().clear()

    def __update(self):
        def additem(item):
            if item.scene() is not self.scene():
                self.addItem(item)

        def removeitem(item):
            if item.scene() is self.scene():
                self.removeItem(item)

        if self.__data is not None:
            additem(self.__cuthigh)
            additem(self.__cutlow)
            additem(self.__tailhigh)
            additem(self.__taillow)

            _, edges = self.__data
            # Update the allowable cutoff line bounds
            minx, maxx = np.min(edges), np.max(edges)
            span = maxx - minx
            bounds = minx - span * 0.005, maxx + span * 0.005

            self.__cuthigh.setBounds(bounds)
            self.__cutlow.setBounds(bounds)

            self.__update_cutlines()
            self.__update_tails()
        else:
            removeitem(self.__cuthigh)
            removeitem(self.__cutlow)
            removeitem(self.__tailhigh)
            removeitem(self.__taillow)

    def __update_cutlines(self):
        self.__cuthigh.setVisible(self.__mode & Histogram.High)
        self.__cuthigh.setValue(self.__max)
        self.__cutlow.setVisible(self.__mode & Histogram.Low)
        self.__cutlow.setValue(self.__min)

    def __update_tails(self):
        if self.__mode == Histogram.NoSelection:
            return
        if self.__data is None:
            return

        hist, edges = self.__data

        self.__taillow.setVisible(self.__mode & Histogram.Low)
        if self.__min > edges[0]:
            datalow = histogram_cut(hist, edges, edges[0], self.__min)
            self.__taillow.setData(*datalow, fillLevel=0, stepMode=True)
        else:
            self.__taillow.clear()

        self.__tailhigh.setVisible(self.__mode & Histogram.High)
        if self.__max < edges[-1]:
            datahigh = histogram_cut(hist, edges, self.__max, edges[-1])
            self.__tailhigh.setData(*datahigh, fillLevel=0, stepMode=True)
        else:
            self.__tailhigh.clear()

    def __on_cuthigh_changed(self):
        self.setUpper(self.__cuthigh.value())

    def __on_cutlow_changed(self):
        self.setLower(self.__cutlow.value())
