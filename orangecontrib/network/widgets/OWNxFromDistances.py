from typing import Optional

import numpy as np
import scipy.sparse as sp
import pyqtgraph as pg

from AnyQt.QtCore import QLineF, QSize, Qt, Signal, QEvent
from AnyQt.QtGui import QDoubleValidator, QIntValidator, QPalette
from AnyQt.QtWidgets import QApplication, QLineEdit

from Orange.data import Table
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.widgets import gui, widget, settings
from Orange.widgets.visualize.utils.plotutils import AxisItem
from Orange.widgets.widget import Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.network.network import Network
from orangecontrib.network.network.base import UndirectedEdges, DirectedEdges
# This enables summarize in widget preview, pylint: disable=unused-import
import orangecontrib.network.widgets
from orangecontrib.network.widgets.utils import items_from_distmatrix, weights_from_distances


class QIntValidatorWithFixup(QIntValidator):
    def fixup(self, text):
        if text and int(text) > self.top():
            return str(self.top())
        return super().fixup(text)


class OWNxFromDistances(widget.OWWidget):
    name = "Network From Distances"
    description = ('Constructs a network by connecting nodes with distances '
                   'below some threshold.')
    icon = "icons/NetworkFromDistances.svg"
    priority = 6440

    class Inputs:
        distances = Input("Distances", DistMatrix)

    class Outputs:
        network = Output("Network", Network)

    resizing_enabled = True
    want_control_area = False

    # The widget stores `density` as setting, because it is more transferable
    # than `threshold`. Internally, the widget uses `threshold` because it is
    # more accurate.
    density = settings.Setting(20, schemaOnly=True)

    class Warning(widget.OWWidget.Warning):
        large_number_of_nodes = \
            Msg('Large number of nodes/edges; expect slow performance.')

    class Error(widget.OWWidget.Error):
        number_of_edges = Msg('The network is too large ({})')

    def __init__(self):
        super().__init__()
        # Matrix from the input, unmodified
        self.matrix: Optional[np.ndarray] = None
        # True, if the matrix is symmetric
        self.symmetric = False
        # All relevant thresholds, that is, all unique values in the distance
        # matrix outside of diagonal. Zero is not explicitly prepended, but
        # will be included if it appears in the distance matrix.
        self.thresholds: Optional[np.ndarray] = None
        # Cumulative frequencies of the thresholds. No prepended zero.
        self.cumfreqs: Optional[np.ndarray] = None
        # Current threshold. Whatever the user sets (threshold, edges, density)
        # will be converted to threshold and stored here.
        self.threshold = 0
        # Number of nodes and edges; use in reports and to set line edits
        self.graph_stat: Optional[tuple[float, float]] = None

        box = gui.vBox(self.mainArea, box=True)
        self.histogram = Histogram(self)
        self.histogram.thresholdChanged.connect(self._on_threshold_dragged)
        self.histogram.draggingFinished.connect(self._on_threshold_drag_finished)
        box.layout().addWidget(self.histogram)

        hbox = gui.hBox(box)
        gui.rubber(hbox)

        _edit_args = dict(alignment=Qt.AlignRight, maximumWidth=50)
        gui.widgetLabel(hbox, "Threshold:")
        self.threshold_edit = QLineEdit(**_edit_args)
        self.threshold_edit.setValidator(QDoubleValidator())
        self.threshold_edit.editingFinished.connect(self._on_threshold_edit)
        hbox.layout().addWidget(self.threshold_edit)
        gui.rubber(hbox)

        gui.widgetLabel(hbox, "Number of edges:")
        self.edges_edit = QLineEdit(**_edit_args)
        self.edges_edit.setValidator(QIntValidatorWithFixup())
        self.edges_edit.editingFinished.connect(self._on_edges_edit)
        hbox.layout().addWidget(self.edges_edit)
        gui.rubber(hbox)

        gui.widgetLabel(hbox, "Density (%):")
        self.density_edit = QLineEdit(**_edit_args)
        self.density_edit.setValidator(QIntValidatorWithFixup(0, 100))
        self.density_edit.editingFinished.connect(self._on_density_edit)
        hbox.layout().addWidget(self.density_edit)
        gui.rubber(hbox)

        self._set_controls_enabled(False)

    def sizeHint(self):  # pragma: no cover
        return QSize(600, 500)

    @property
    def eff_distances(self):
        n = len(self.matrix)
        return n * (n - 1) // (1 + self.symmetric)

    def _on_threshold_dragged(self, threshold):
        self.threshold = threshold
        self.update_edits(from_dragging=True)

    def _on_threshold_drag_finished(self):
        self.generate_network()

    def _on_threshold_edit(self):
        self.threshold = float(self.threshold_edit.text())
        self.generate_network()
        self.update_histogram_lines()
        self.threshold_edit.selectAll()

    def _on_density_edit(self):
        self.density = int(self.density_edit.text())
        self.set_threshold_from_density()
        self.generate_network()
        self.update_histogram_lines()
        self.density_edit.selectAll()

    def _on_edges_edit(self):
        edges = int(self.edges_edit.text())
        self.set_threshold_from_edges(edges)
        self.generate_network()
        self.update_histogram_lines()
        self.edges_edit.selectAll()

    def set_threshold_from_density(self):
        self.set_threshold_from_edges(
            int(np.ceil(self.density * self.eff_distances / 100)))

    def set_threshold_from_edges(self, edges):
        # Set the threshold that will give at least the given number of edges
        if edges == 0:
            self.threshold = 0
        else:
            matrix = self.matrix
            if not self.symmetric:
                matrix = self.__no_diagonal(matrix)
            thresholds = np.sort(matrix.flat)
            edges = min(edges, len(thresholds)) or 1
            self.threshold = thresholds[edges - 1]

    def edges_from_threshold(self):
        """
        Fast, histogram-based estimate of the number of edges below
        the current threshold.
        """
        idx = np.searchsorted(self.thresholds, self.threshold, side='right')
        return self.cumfreqs[idx - 1] if idx else 0

    def update_edits(self, from_dragging=False):
        n_decimals = max(0, -int(np.floor(np.log10(np.max(self.thresholds)))) + 2)
        if from_dragging or self.graph_stat is None:
            edges = self.edges_from_threshold()
        else:
            _, edges = self.graph_stat
        self.density = int(round(100 * edges / self.eff_distances))

        appx = "~" if from_dragging else ""
        self.threshold_edit.setText(f"{self.threshold:.{n_decimals}f}")
        self.edges_edit.setText(appx + str(edges))
        self.density_edit.setText(appx + str(self.density))

    def update_histogram_lines(self):
        if self.graph_stat is None:
            return
        _, edges = self.graph_stat
        self.histogram.update_region(self.threshold, True, True, edges=edges)

    def _set_controls_enabled(self, enabled):
        for edit in (self.threshold_edit, self.edges_edit, self.density_edit):
            edit.setEnabled(enabled)

    # This can be removed when DistMatrix.flat is fixed to include this code
    @staticmethod
    def __no_diagonal(matrix):
        return np.lib.stride_tricks.as_strided(
            matrix.reshape(matrix.size, -1)[1:],
            shape=(matrix.shape[0] - 1, matrix.shape[1]),
            strides=(matrix.strides[0] + matrix.strides[1],
                     matrix.strides[1]),
            writeable=False
        )

    @Inputs.distances
    def set_matrix(self, matrix: DistMatrix):
        if matrix is not None and matrix.size <= 1:
            matrix = None

        self.matrix = matrix
        if matrix is None:
            self.thresholds = None
            self.symmetric = True
            self._set_controls_enabled(False)
            self.histogram.clear_graph()
            self.generate_network()
            return

        self._set_controls_enabled(True)
        self.symmetric = matrix.is_symmetric()
        if not self.symmetric:
            matrix = self.__no_diagonal(matrix)

        if self.eff_distances < 1000:
            self.thresholds, freq = np.unique(matrix.flat, return_counts=True)
        else:
            freq, edges = np.histogram(matrix.flat, bins=1000)
            self.thresholds = edges[1:]
        self.cumfreqs = np.cumsum(freq)
        self.histogram.set_graph(self.thresholds, self.cumfreqs)

        self.edges_edit.validator().setRange(0, self.eff_distances)
        self.set_threshold_from_density()
        self.generate_network()
        self.update_histogram_lines()

    def generate_network(self):
        self.Error.clear()
        self.Warning.clear()
        matrix = self.matrix

        if matrix is None:
            self.graph_stat = None
            self.Outputs.network.send(None)
            return

        nedges = self.edges_from_threshold()
        if nedges > 200000:
            self.Error.number_of_edges(nedges)
            self.graph_stat = None
            self.Outputs.network.send(None)
            return

        mask = np.array(matrix <= self.threshold)
        if self.symmetric:
            mask &= np.tri(*matrix.shape, k=-1, dtype=bool)
        else:
            mask &= ~np.eye(*matrix.shape, dtype=bool)
        if np.sum(mask):
            # Set the weights so that the edge with the smallest distances
            # have the weight of 1, and the edges with the largest distances
            # have the weight of 0.01; the rest are scaled logarithmically
            weights = matrix[mask].astype(float)
            weights = weights_from_distances(weights)
            edges = sp.csr_matrix((weights, mask.nonzero()), shape=matrix.shape)
        else:
            edges = sp.csr_matrix(matrix.shape)
        edge_type = UndirectedEdges if self.symmetric else DirectedEdges
        graph = Network(items_from_distmatrix(self.matrix), edge_type(edges))
        nodes, edges = graph.number_of_nodes(), graph.number_of_edges()
        self.graph_stat = nodes, edges
        self.Warning.large_number_of_nodes(shown=nodes > 3000 or edges > 10000)
        self.Outputs.network.send(graph)
        self.update_edits()

    def send_report(self):
        # We take the threshold from the edit box to have the same number of
        # decimals (the user may have entered a value with more decimals than
        # we'd set within update_edits
        if self.graph_stat is None:
            return

        self.report_items("Settings", [
            ("Threshold", self.threshold_edit.text()),
            ("Density", self.density_edit.text()),
            ("Edges", self.edges_edit.text()),
        ])

        self.report_name("Histogram")
        self.report_plot(self.histogram)

        nodes, edges = self.graph_stat
        self.report_items(
            "Output network",
            [("Vertices", len(self.matrix)),
             ("Edges", f"{edges} "
                       f"({edges / nodes:.2f} per vertex), "
                       f"density: {edges / self.eff_distances:.2%}")])


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

    def hoverEvent(self, ev):  # pragma: no cover
        if ev.isEnter():
            QApplication.setOverrideCursor(
                Qt.SizeVerCursor if self.angle == 0 else Qt.SizeHorCursor)
        elif ev.isExit():
            QApplication.restoreOverrideCursor()
        return super().hoverEvent(ev)

# Patched so that the Histogram's LinearRegionItem works on MacOS
pg.InfiniteLine = InfiniteLine
pg.graphicsItems.LinearRegionItem.InfiniteLine = InfiniteLine


class Histogram(pg.PlotWidget):
    thresholdChanged = Signal(float)
    draggingFinished = Signal()

    def __init__(self, parent, **kwargs):
        axisItems = kwargs.pop("axisItems", None)
        if axisItems is None:
            axisItems = {"left": AxisItem("left"), "bottom": AxisItem("bottom")}
        super().__init__(
            parent, setAspectLocked=True, background=None, axisItems=axisItems,
            **kwargs)
        self.setBackgroundRole(QPalette.Base)
        self.setPalette(QPalette())
        self.__updateScenePalette()

        self.curve = self.plot([0, 1], [0, 0], pen=pg.mkPen('b', width=3))
        self.fill_curve = self.plotItem.plot([0, 1], [0, 0],
            fillLevel=0, pen=pg.mkPen('b', width=4), brush='#02f3')
        self.plotItem.setContentsMargins(12, 12, 12, 12)
        self.plotItem.vb.setMouseEnabled(x=False, y=False)
        # Add another left axis
        self.prop_axis = prop_axis = AxisItem(orientation='right')
        self.plotItem.layout.addItem(prop_axis, 2, 3)
        prop_axis.linkToView(self.plotItem.vb)
        # Set axes titles
        self.plotItem.setLabel('bottom', "Threshold Distance")
        self.plotItem.setLabel('left', "Number of Edges")
        prop_axis.setLabel("Edge Density", units="%")

        line_args = dict(
            movable=True,
            pen=pg.mkPen('k', width=1, style=Qt.DashLine),
            hoverPen=pg.mkPen('k', width=2, style=Qt.DashLine)
        )
        self.hline = pg.InfiniteLine(angle=0, **line_args)
        self.plotItem.addItem(self.hline, ignoreBounds=True)
        self.hline.sigDragged.connect(self._hline_dragged)
        self.hline.sigPositionChangeFinished.connect(self.draggingFinished)

        self.vline = pg.InfiniteLine(angle=90, **line_args)
        self.plotItem.addItem(self.vline, ignoreBounds=True)
        self.vline.sigDragged.connect(self._vline_dragged)
        self.vline.sigPositionChangeFinished.connect(self.draggingFinished)

        self.clear_graph()

    def setScene(self, scene):  # pragma: no cover
        super().setScene(scene)
        self.__updateScenePalette()

    def __updateScenePalette(self):  # pragma: no cover
        scene = self.scene()
        if scene is not None:
            scene.setPalette(self.palette())

    def changeEvent(self, event):  # pragma: no cover
        if event.type() == QEvent.PaletteChange:
            self.__updateScenePalette()
            self.resetCachedContent()
        super().changeEvent(event)

    def update_region(self, thresh,
                      set_hline=False, set_vline=False,
                      edges=None):
        xData, yData = self.curve.xData, self.curve.yData
        high = np.searchsorted(xData, thresh, side='right')
        self.fill_curve.setData(xData[:high], yData[:high])
        if set_hline:
            if edges is None:
                if high == len(yData):
                    edges = yData[-1]
                else:
                    edges = yData[high - (xData[high] > thresh)]
            self.hline.setPos(edges)
        if set_vline:
            self.vline.setPos(thresh)

    def _hline_dragged(self):
        xData, yData = self.curve.xData, self.curve.yData
        pos = self.hline.value()
        idx = np.searchsorted(yData, pos, side='left')
        thresh = xData[min(idx, len(xData) - 1)]
        self.update_region(thresh, set_vline=True)
        self.thresholdChanged.emit(thresh)

    def _vline_dragged(self):
        thresh = self.vline.value()
        self.update_region(thresh, set_hline=True)
        self.thresholdChanged.emit(thresh)

    def _elements(self):
        return self.curve, self.fill_curve, self.hline, self.vline

    def clear_graph(self):
        for el in self._elements():
            el.hide()
        self.prop_axis.setScale(1)

    def set_graph(self, edges, cumfreqs):
        self.curve.setData(np.hstack(([0], edges)),
                           np.hstack(([0], cumfreqs)))
        self.getAxis('left').setRange(0, cumfreqs[-1])
        self.hline.setBounds([0, cumfreqs[-1]])
        self.prop_axis.setScale(1 / cumfreqs[-1] * 100)
        self.getAxis('bottom').setRange(0, edges[-1])
        self.vline.setBounds([0, edges[-1]])
        self.update_region(edges[0], set_hline=True, set_vline=True)
        for el in self._elements():
            el.show()


if __name__ == "__main__":
    distances5 = DistMatrix(np.array([[0., 1, 2, 5, 10],
                                      [1, -1, 5, 5, 13],
                                      [2, 5, 2, 6, 13],
                                      [5, 5, 6, 3, 15],
                                      [10, 13, 13, 15, 0]]))
    distances = Euclidean(Table("iris"))
    WidgetPreview(OWNxFromDistances).run(set_matrix=distances)
