import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse as sp

from AnyQt.QtCore import Qt

from orangewidget.report import report
from Orange.data import Table
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.network.network import Network
from orangecontrib.network.network.base import UndirectedEdges, DirectedEdges
# This enables summarize in widget preview, pylint: disable=unused-import
import orangecontrib.network.widgets
from orangecontrib.network.widgets.utils import items_from_distmatrix, weights_from_distances


class OWNxNeighbor(widget.OWWidget):
    name = "Network of Neighbors"
    description = 'Constructs a network by connecting nearest neighbors.'
    icon = "icons/NetworkOfNeighbors.svg"
    priority = 6445

    class Inputs:
        distances = Input("Distances", DistMatrix)

    class Outputs:
        network = Output("Network", Network)

    resizing_enabled = False
    want_main_area = False

    k = settings.Setting(3)
    directed = settings.Setting(False)
    auto_apply = settings.Setting(True)

    def __init__(self):
        super().__init__()
        self.matrix = None
        self.symmetric = False
        self.graph = None

        box = gui.vBox(self.controlArea, box=True)
        gui.spin(
            box, self, 'k', 1, 20, label="Number of neighbors:",
            alignment=Qt.AlignRight, callback=self.commit.deferred,
            controlWidth=60)
        gui.checkBox(
            box, self, 'directed', 'Directed edges',
            callback=self.commit.deferred)

        gui.auto_apply(self.controlArea, self)

    @Inputs.distances
    def set_matrix(self, matrix: DistMatrix):
        if matrix is not None and matrix.size < 2:
            matrix = None
        if matrix is None:
            self.symmetric = False
            self.controls.k.setMaximum(1)
        else:
            self.controls.k.setMaximum(min(20, matrix.shape[0] - 1))
            self.symmetric = matrix.is_symmetric()
        self.matrix = matrix
        self.commit.now()

    @gui.deferred
    def commit(self):
        if self.matrix is None:
            self.graph = None
        else:
            edge_type = DirectedEdges if self.directed else UndirectedEdges
            items = items_from_distmatrix(self.matrix)
            edges = edge_type(self.get_neighbors().tocsr())
            self.graph = Network(items, edges)
        self.Outputs.network.send(self.graph)

    def get_neighbors(self):
        k = min(self.k, self.matrix.shape[0] - 1)
        matrix = np.asarray(self.matrix)
        if matrix.base is not None:
            matrix = matrix.copy()
        # TODO: use matrix.diag.fill(np.inf) when we require a newer numpy
        matrix[np.eye(len(matrix), dtype=bool)] = np.inf
        cols = np.argsort(matrix, axis=1)[:, :k].flatten().astype(np.int32)
        rows = np.repeat(np.arange(len(matrix), dtype=np.int32), k)
        if not self.directed:
            edges = np.array([rows, cols]).T
            edges.sort()
            cols, rows = np.unique(edges, axis=0).T

        weights = weights_from_distances(matrix[rows, cols])
        mask = weights > 0
        weights = weights[mask]

        return sp.coo_array((weights, (rows, cols)), matrix.shape)

    def send_report(self):
        self.report_items("Settings", [
            ("Neighbors", self.k),
            ("Directed", report.bool_str(self.directed))
        ])
        if self.graph is not None:
            self.report_items("Output network", [
                ("Vertices", self.graph.number_of_nodes()),
                ("Edges", self.graph.number_of_edges())])


if __name__ == "__main__":
    distances = Euclidean(Table("iris"))
    WidgetPreview(OWNxNeighbor).run(set_matrix=distances)
