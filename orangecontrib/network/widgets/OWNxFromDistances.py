import copy
import random
import sys

import Orange
from Orange.widgets import gui, widget
import orangecontrib.network as network

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from orangecontrib.network.widgets.OWNxHist import *


class OWNxFromDistances(widget.OWWidget, OWNxHist):
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

    settingsList=["spinLowerThreshold", "spinUpperThreshold", "netOption",
                  "dstWeight", "kNN", "percentil", "andor", "excludeLimit"]
    # TODO: set settings

    def __init__(self):
        super().__init__()
        OWNxHist.__init__(self)

        # GUI
        # general settings
        boxHistogram = gui.widgetBox(self.mainArea, box = "Distance histogram")
        self.histogram = Histogram(self)
        boxHistogram.layout().addWidget(self.histogram)
        self.addHistogramControls()

        boxHistogram.setMinimumWidth(500)
        boxHistogram.setMinimumHeight(300)

        # info
        boxInfo = gui.widgetBox(self.controlArea, box = "Info")
        self.infoa = gui.widgetLabel(boxInfo, "No data loaded.")
        self.infob = gui.widgetLabel(boxInfo, '')
        self.infoc = gui.widgetLabel(boxInfo, '')

        gui.rubber(self.controlArea)

        self.resize(700, 100)

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Edge thresholds", "%.5f - %.5f" % \
                              (self.spinLowerThreshold, \
                               self.spinUpperThreshold)),
                             ("Selected vertices", ["All", \
                                "Without isolated vertices",
                                "Largest component",
                                "Connected with vertex"][self.netOption]),
                             ("Weight", ["Distance", "1 - Distance"][self.dstWeight])])
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

if __name__ == "__main__":
    appl = QApplication(sys.argv)
    ow = OWNxFromDistances()
    ow.show()
    appl.exec_()
