import numpy as np
import scipy.sparse as sp
import pyqtgraph as pg

from AnyQt.QtCore import QLineF, QSize, Qt, Signal, QEvent
from AnyQt.QtGui import QDoubleValidator, QIntValidator, QPalette

from Orange.data import Domain, StringVariable, Table
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


class QIntValidatorWithFixup(QIntValidator):
    def fixup(self, text):
        if text and int(text) > self.top():
            return str(self.top())
        return super().fixup(text)


class OWNxFromDistances(widget.OWWidget):
    name = "Network From Distances"
    description = ('Constructs a network by connecting nodes with distances '
                   'below somethreshold.')
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
    density = settings.Setting(20)

    class Warning(widget.OWWidget.Warning):
        large_number_of_nodes = \
            Msg('Large number of nodes/edges; expect slow performance.')

    class Error(widget.OWWidget.Error):
        number_of_edges = Msg('The network is too large ({})')

    def __init__(self):
        super().__init__()
        self.matrix = None
        self.symmetric = False
        self.threshold = 0
        self.graph = None

        box = gui.vBox(self.mainArea, box=True)
        self.histogram = Histogram(self)
        self.histogram.thresholdChanged.connect(self._on_threshold_dragged)
        self.histogram.draggingFinished.connect(self._on_threshold_drag_finished)
        box.layout().addWidget(self.histogram)

        hbox = gui.hBox(box)
        gui.rubber(hbox)

        _edit_args = dict(
            orientation=Qt.Horizontal, alignment=Qt.AlignRight, controlWidth=50)
        self.labels = []
        self.threshold_label = gui.widgetLabel(hbox, "Threshold:")
        self.labels.append(self.threshold_label)
        self.threshold_edit = gui.lineEdit(
            hbox, self, '',
            validator=QDoubleValidator(), callback=self._on_threshold_edit,
            **_edit_args)
        gui.rubber(hbox)

        self.edges_label = gui.widgetLabel(hbox, "Number of edges:")
        self.edges_edit = gui.lineEdit(
            hbox, self, '',
            validator=QIntValidatorWithFixup(), callback=self._on_edges_edit,
            **_edit_args)
        self.labels.append(self.edges_label)
        gui.rubber(hbox)

        self.density_label = gui.widgetLabel(hbox, "Density (%):")
        self.density_edit = gui.lineEdit(
            hbox, self, 'density',
            validator=QIntValidatorWithFixup(0, 100),
            callback=self._on_density_edit,
            **_edit_args)
        self.labels.append(self.density_label)
        gui.rubber(hbox)

    def sizeHint(self):
        return QSize(600, 500)

    @property
    def eff_distances(self):
        n = len(self.matrix)
        return n * (n - 1) // (1 + self.symmetric)

    def _on_threshold_dragged(self, threshold):
        self.threshold = threshold
        self.update_edits(self.histogram)

    def _on_threshold_drag_finished(self):
        self.generate_network()

    def _on_threshold_edit(self):
        self.threshold = float(self.threshold_edit.text())
        self.update_edits(self.threshold_label)
        self.generate_network()

    def _on_density_edit(self):
        self.density = int(self.density_edit.text())
        self.set_threshold_from_density()
        self.update_edits(self.density_label)
        self.generate_network()

    def _on_edges_edit(self):
        edges = int(self.edges_edit.text())
        self.set_threshold_from_edges(edges)
        self.update_edits(self.edges_label)
        self.generate_network()

    def set_threshold_from_density(self):
        self.set_threshold_from_edges(self.density * self.eff_distances / 100)

    def set_threshold_from_edges(self, edges):
        # Set the threshold that will give at least the given number of edges
        if edges == 0:
            self.threshold = 0
        else:
            self.threshold = self.edges[np.searchsorted(self.cumfreqs, edges)]

    def edges_from_threshold(self):
        idx = np.searchsorted(self.edges, self.threshold, side='right')
        return self.cumfreqs[idx - 1] if idx else 0

    def update_edits(self, reference):
        if reference is not self.threshold_label:
            self.threshold_edit.setText(f"{self.threshold:.2f}")
        if reference is not self.edges_label:
            edges = self.edges_from_threshold()
            self.edges_edit.setText(str(edges))
        if reference is not self.density_label:
            self.density = \
                int(round(100 * self.edges_from_threshold() / self.eff_distances))
            self.density_edit.setText(str(self.density))
        if reference is not self.histogram:
            self.histogram.update_region(self.threshold, True, True,
                                         density=self.density)

        for label in self.labels:
            font = label.font()
            font.setBold(label is reference)
            label.setFont(font)

    @Inputs.distances
    def set_matrix(self, matrix: DistMatrix):
        if matrix is not None and not matrix.size:
            matrix = None

        self.matrix = matrix
        if matrix is None:
            self.symmetric = True
            self.histogram.set_values([], [])
            self.generate_network()
            return

        self.symmetric = matrix.is_symmetric()
        # This can be removed when DistMatrix.flat is fixed to include this code
        if not self.symmetric:
            matrix = np.lib.stride_tricks.as_strided(
                matrix.reshape(matrix.size, -1)[1:],
                shape=(matrix.shape[0] - 1, matrix.shape[1]),
                strides=(matrix.strides[0] + matrix.strides[1],
                         matrix.strides[1]),
                writeable=False
            )

        if self.eff_distances < 1000:
            self.edges, freq = np.unique(matrix.flat, return_counts=True)
        else:
            freq, edges = np.histogram(matrix.flat, bins=1000)
            self.edges = edges[:-1]
        self.cumfreqs = np.cumsum(freq)
        self.histogram.set_values(self.edges, self.cumfreqs)

        self.edges_edit.validator().setRange(0, self.eff_distances)
        self.set_threshold_from_density()
        self.update_edits(self.density_label)
        self.generate_network()

    def generate_network(self):
        self.Error.clear()
        self.Warning.clear()
        matrix = self.matrix

        if matrix is None:
            self.graph = None
            self.Outputs.network.send(None)
            return

        threshold = float(self.threshold_edit.text())
        nedges = self.edges_from_threshold()
        if nedges > 200000:
            self.Error.number_of_edges(nedges)
            self.graph = None
            self.Outputs.network.send(None)
            return

        mask = np.array(matrix <= threshold)
        if self.symmetric:
            mask &= np.tri(*matrix.shape, k=-1, dtype=bool)
        else:
            mask &= ~np.eye(*matrix.shape, dtype=bool)
        if np.sum(mask):
            # Set the weights so that the edge with the smallest distances
            # have the weight of 1, and the edges with the largest distances
            # have the weight of 0.01; the rest are scaled logarithmically
            weights = matrix[mask].astype(float)
            mi, ma = np.min(weights), np.max(weights)
            if ma - mi < 1e-6:
                weights.fill(1)
            else:
                a = np.log(199) / (ma - mi)
                weights -= mi
                weights *= a
                np.exp(weights, out=weights)
                weights += 1
                np.reciprocal(weights, out=weights)
                weights *= 2
            edges = sp.csr_matrix((weights, mask.nonzero()), shape=matrix.shape)
        else:
            edges = sp.csr_matrix(matrix.shape)
        edge_type = UndirectedEdges if self.symmetric else DirectedEdges
        self.graph = Network(self._items_for_network(), edge_type(edges))
        self.Warning.large_number_of_nodes(
            shown=self.graph.number_of_nodes() > 3000
            or self.graph.number_of_edges() > 10000)

        self.Outputs.network.send(self.graph)

    def _items_for_network(self):
        assert self.matrix is not None

        if self.matrix.row_items is not None:
            row_items = self.matrix.row_items
            if isinstance(row_items, Table):
                if self.matrix.axis == 1:
                    items = row_items
                else:
                    items = [[v.name] for v in row_items.domain.attributes]
            else:
                items = [[str(x)] for x in self.matrix.row_items]
        else:
            items = [[str(i)] for i in range(1, 1 + self.matrix.shape[0])]
        if not isinstance(items, Table):
            items = Table.from_list(
                Domain([], metas=[StringVariable('label')]),
                items)
        return items

    def send_report(self):
        self.report_items("Settings", [
            ("Threshold", self.threshold),
            ("Density", self.density),
            ("Edges", self.edges_from_threshold()),
        ])

        if self.graph is None:
            return

        self.report_name("Histogram")
        self.report_plot(self.histogram)

        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()
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

    def setScene(self, scene):
        super().setScene(scene)
        self.__updateScenePalette()

    def __updateScenePalette(self):
        scene = self.scene()
        if scene is not None:
            scene.setPalette(self.palette())

    def changeEvent(self, event):
        if event.type() == QEvent.PaletteChange:
            self.__updateScenePalette()
            self.resetCachedContent()
        super().changeEvent(event)

    def update_region(self, thresh,
                      set_hline=False, set_vline=False,
                      density=None):
        high = np.searchsorted(self.xData, thresh, side='right')
        self.fill_curve.setData(self.xData[:high], self.yData[:high])
        if set_hline:
            if density is None:
                density = self.yData[min(high, len(self.yData) - 1)]
            self.hline.setPos(density)
        if set_vline:
            self.vline.setPos(thresh)

    def _hline_dragged(self):
        pos = self.hline.value()
        idx = np.searchsorted(self.yData, pos, side='right')
        thresh = self.xData[min(idx, len(self.xData) - 1)]
        self.update_region(thresh, set_vline=True)
        self.thresholdChanged.emit(thresh)

    def _vline_dragged(self):
        thresh = self.vline.value()
        self.update_region(thresh, set_hline=True)
        self.thresholdChanged.emit(thresh)

    def set_values(self, edges, cumfreqs):
        self.fill_curve.setData([0, 1], [0, 0])
        if not len(edges):
            self.curve.setData([0], [1])
            self.fill_curve.setData([0], [1])
            return
        self.curve.setData(np.hstack(([0], edges)),
                           np.hstack(([0], cumfreqs)))
        self.getAxis('left').setRange(0, cumfreqs[-1])
        self.hline.setBounds([0, cumfreqs[-1]])
        self.prop_axis.setScale(1 / cumfreqs[-1] * 100)
        self.getAxis('bottom').setRange(edges[0], edges[-1])
        self.vline.setBounds([edges[0], edges[-1]])
        self.update_region(edges[0], set_hline=True, set_vline=True)

    @property
    def xData(self):
        return self.curve.xData

    @property
    def yData(self):
        return self.curve.yData


if __name__ == "__main__":
    distances5 = DistMatrix(np.array([[0., 1, 2, 5, 10],
                                      [1, -1, 5, 5, 13],
                                      [2, 5, 2, 6, 13],
                                      [5, 5, 6, 3, 15],
                                      [10, 13, 13, 15, 0]]))
    distances = Euclidean(Table("iris"))
    WidgetPreview(OWNxFromDistances).run(set_matrix=distances)
