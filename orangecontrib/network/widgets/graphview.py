import numpy as np
import pyqtgraph as pg
import time

from Orange.util import scale
from Orange.widgets.settings import Setting
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase


class GraphView(OWScatterPlotBase):
    showEdgeWeights = Setting(False)
    relativeEdgeWidths = Setting(False)
    edge_width = Setting(2)

    def __init__(self, master, parent=None):
        super().__init__(master)
        self.paired_indices = None
        self.pairs_curve = None
        self.scatterplot_marked = None
        self.last_click = (-1, None)

    def clear(self):
        super().clear()
        self.paired_indices = None
        self.pairs_curve = None
        self.scatterplot_marked = None
        self.last_click = (-1, None)

    def update_coordinates(self):
        self.update_edges()
        x, y = self.get_coordinates()
        if x is None:
            return
        if self.pairs_curve is None:
            self.scatterplot_marked = pg.ScatterPlotItem([], [])
            self.plot_widget.addItem(self.scatterplot_marked)
            edges = self.master.get_edges()
            srcs, dests = edges.row, edges.col
            n_edges = len(srcs)
            self.paired_indices = np.empty((2 * n_edges, ), dtype=int)
            self.paired_indices[::2] = srcs
            self.paired_indices[1::2] = dests
            self.pairs_curve = pg.PlotCurveItem(
                x[self.paired_indices], y[self.paired_indices],
                pen=self._edge_curve_pen(),
                connect="pairs", antialias=True)
            self.plot_widget.addItem(self.pairs_curve)
        else:
            self.pairs_curve.setData(
                x[self.paired_indices], y[self.paired_indices],
                pen=self._edge_curve_pen(),
                connect="pairs", antialias=True)

        super().update_coordinates()
        self.update_marks()

    """
            if not self.network: return
        if self.graph.relativeEdgeWidths:
            widths = [self.network.adj[u][v].get('weight', 1)
                      for u, v in self.network.edges()]
            widths = scale(widths, .7, 8) * np.log2(self.edge_width/4 + 1)
        else:
            widths = (.7 * self.edge_width for _ in range(self.network.number_of_edges()))
#        for edge, width in zip(self.view.edges, widths):
 #           edge.setSize(width)
    """
    def update_edges(self):
        pass

    def set_edge_pen(self):
        if self.pairs_curve is not None:
            self.pairs_curve.setPen(self._edge_curve_pen())

    def _edge_curve_pen(self):
        return pg.mkPen(0.5 if self.class_density else 0.8,
                        width=self.edge_width, cosmetic=True)

    def set_edge_sizes(self):
        self.set_edge_pen()

    def update_density(self):
        super().update_density()
        self.set_edge_pen()

    def update_marks(self):
        if not self.scatterplot_item:
            return

        marked = self.master.get_marked_nodes()
        if marked is None:
            self.scatterplot_marked.setData([], [])
            return

        x, y = self.scatterplot_item.getData()
        self.scatterplot_marked.setData(
            x[marked], y[marked], size=25,
            pen=pg.mkPen(None), brush=pg.mkBrush("aff"))

    def select_by_click(self, _, points):
        # Poor man's double click
        indices = [p.data() for p in points]
        last_time, last_indices = self.last_click
        if time.time() - last_time < 0.25 and indices == last_indices:
            indices = self.master.get_reachable(indices)
        self.last_click = (time.time(), indices)
        self.select_by_indices(indices)
