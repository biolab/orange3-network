import numpy as np
import pyqtgraph as pg
import time

from AnyQt.QtCore import QLineF
from AnyQt.QtGui import QPen, QBrush

from Orange.util import scale
from Orange.widgets.settings import Setting
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase


class PlotVarWidthCurveItem(pg.PlotCurveItem):
    def __init__(self, *args, **kwargs):
        self.widths = kwargs.pop("widths", None)
        self.setPen(kwargs.pop("pen", None))
        super().__init__(*args, **kwargs)

    def setWidths(self, widths):
        self.widths = widths
        self.update()

    def setPen(self, pen):
        self.pen = QPen(pen)
        self.pen.setCosmetic(True)

    def setData(self, *args, **kwargs):
        self.widths = kwargs.pop("widths", self.widths)
        super().setData(*args, **kwargs)

    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0 or self.widths is None:
            return
        p.setRenderHint(p.Antialiasing, True)
        p.setCompositionMode(p.CompositionMode_SourceOver)
        for x0, y0, x1, y1, w in zip(self.xData[::2], self.yData[::2],
                                     self.xData[1::2], self.yData[1::2],
                                     self.widths):
            self.pen.setWidth(w)
            p.setPen(self.pen)
            p.drawLine(QLineF(x0, y0, x1, y1))


class GraphView(OWScatterPlotBase):
    show_edge_weights = Setting(False)
    relative_edge_widths = Setting(False)
    edge_width = Setting(2)

    COLOR_NOT_SUBSET = (255, 255, 255, 255)
    COLOR_SUBSET = (0, 0, 0, 255)
    COLOR_DEFAULT = (255, 255, 255, 0)

    class Simplifications:
        SameEdgeWidth, NoEdges, NoDensity, NoLabels = 1, 2, 4, 8
        NoSimplifications, All = 0, 255

    def __init__(self, master, parent=None):
        super().__init__(master)
        self._reset_attributes()
        self.simplify = self.Simplifications.NoSimplifications

    def clear(self):
        super().clear()
        self._reset_attributes()

    def _reset_attributes(self):
        self.paired_indices = None
        self.edge_curve = None
        self.scatterplot_marked = None
        self.last_click = (-1, None)

    def update_coordinates(self):
        super().update_coordinates()
        self.update_marks()
        self.update_edges()

    def set_simplifications(self, simplifications):
        S = self.Simplifications
        self.plot_widget.setUpdatesEnabled(False)
        if (self.simplify ^ simplifications) & S.SameEdgeWidth:
            self.simplify ^= S.SameEdgeWidth
            self._remove_edges()
            if not self.simplify & simplifications & S.NoEdges \
                    and not self.simplify & S.SameEdgeWidth:
                self.update_edges()
        for flag, remove, update in (
                (S.NoDensity, self._remove_density, self.update_density),
                (S.NoLabels, self._remove_labels, self.update_labels),
                (S.NoEdges, self._remove_edges, self.update_edges)):
            if simplifications & flag != self.simplify & flag:
                if simplifications & flag:
                    self.simplify += flag
                    remove()
                else:
                    self.simplify -= flag
                    update()
        self.plot_widget.setUpdatesEnabled(True)

    def update_edges(self):
        if not self.scatterplot_item \
                or self.simplify & self.Simplifications.NoEdges:
            return
        x, y = self.scatterplot_item.getData()
        edges = self.master.get_edges()
        srcs, dests, weights = edges.row, edges.col, edges.data
        if self.edge_curve is None:
            self.paired_indices = np.empty((2 * len(srcs), ), dtype=int)
            self.paired_indices[::2] = srcs
            self.paired_indices[1::2] = dests

        kwargs = dict(x=x[self.paired_indices], y=y[self.paired_indices],
                      pen=self._edge_curve_pen(), antialias=True)
        if self.relative_edge_widths \
                and not self.simplify & self.Simplifications.SameEdgeWidth:
            cls = PlotVarWidthCurveItem
            kwargs['widths'] = \
                scale(weights, .7, 8) * np.log2(self.edge_width / 4 + 1)
        else:
            cls = pg.PlotCurveItem
            kwargs['connect'] = 'pairs'

        if type(self.edge_curve) != cls:  # Check for exact type
            self.plot_widget.removeItem(self.edge_curve)
            self.edge_curve = None
        if self.edge_curve is None:
            self.edge_curve = cls(**kwargs)
            self.plot_widget.addItem(self.edge_curve)
            self._put_nodes_on_top()
        else:
            self.edge_curve.setData(**kwargs)

    def _put_nodes_on_top(self):
        if self.scatterplot_item:
            self.plot_widget.removeItem(self.scatterplot_item_sel)
            self.plot_widget.removeItem(self.scatterplot_item)
            self.plot_widget.addItem(self.scatterplot_item_sel)
            self.plot_widget.addItem(self.scatterplot_item)

    def set_edge_pen(self):
        if self.edge_curve:
            self.edge_curve.setPen(self._edge_curve_pen())

    def _edge_curve_pen(self):
        return pg.mkPen(
            0.5 if self.class_density else 0.8,
            width=self.edge_width,
            cosmetic=True)

    def set_edge_labels(self):
        pass

    def _remove_edges(self):
        if self.edge_curve:
            self.plot_widget.removeItem(self.edge_curve)
            self.edge_curve = None

    def update_density(self):
        if not self.simplify & self.Simplifications.NoDensity:
            super().update_density()
            self.set_edge_pen()

    def _remove_density(self):
        if self.density_img:
            self.plot_widget.removeItem(self.density_img)
            self.density_img = None

    def update_labels(self):
        if self.simplify & self.Simplifications.NoLabels:
            return
        # This is not nice, but let's add methods to the parent just
        # to support this specific case
        if self.label_only_selected and self.scatterplot_item:
            marked = self.master.get_marked_nodes()
            if marked is not None and len(marked):
                if self.selection is None:
                    selection = None
                    self.selection = \
                        np.zeros(len(self.scatterplot_item.data), dtype=bool)
                else:
                    selection = np.array(self.selection)
                self.selection[self.master.get_marked_nodes()] = 1
                super().update_labels()
                self.selection = selection
                return
        super().update_labels()

    def _remove_labels(self):
        for label in self.labels:
            self.plot_widget.removeItem(label)
        self.labels = []

    def update_marks(self):
        if self.scatterplot_marked is None:
            self.scatterplot_marked = pg.ScatterPlotItem([], [])
            self.plot_widget.addItem(self.scatterplot_marked)

        marked = self.master.get_marked_nodes()
        if marked is None:
            self.scatterplot_marked.setData([], [])
            return
        x, y = self.get_coordinates()
        if x is None:  # sanity check; there can be no marked nodes if x is None
            return
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
