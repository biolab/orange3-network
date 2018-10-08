import numpy as np
import pyqtgraph as pg
import time

from Orange.util import scale
from Orange.widgets.settings import Setting
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase


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
        self.edge_curves = []
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
        if self.relative_edge_widths \
                and not self.simplify & self.Simplifications.SameEdgeWidth:
            self.update_edges_relative()
        else:
            self.update_edges_uniform()

    def update_edges_uniform(self):
        if self.edge_curves:
            for edge in self.edge_curves:
                self.plot_widget.removeItem(edge)
            self.edge_curves.clear()

        x, y = self.get_coordinates()
        if x is None:
            return
        if self.edge_curve is None:
            edges = self.master.get_edges()
            srcs, dests = edges.row, edges.col
            n_edges = len(srcs)
            self.paired_indices = np.empty((2 * n_edges, ), dtype=int)
            self.paired_indices[::2] = srcs
            self.paired_indices[1::2] = dests
            self.edge_curve = pg.PlotCurveItem(
                x[self.paired_indices], y[self.paired_indices],
                pen=self._edge_curve_pen(),
                connect="pairs", antialias=True)
            self.plot_widget.addItem(self.edge_curve)
            self._put_nodes_on_top()
        else:
            self.edge_curve.setData(
                x[self.paired_indices], y[self.paired_indices],
                pen=self._edge_curve_pen(),
                connect="pairs", antialias=True)

    def update_edges_relative(self):
        if self.edge_curve:
            self.plot_widget.removeItem(self.edge_curve)
            self.edge_curve = None

        x, y = self.get_coordinates()
        if x is None:
            return
        edges = self.master.get_edges()
        weights, srcs, dests = edges.data, edges.row, edges.col
        widths = scale(weights, .7, 8) * np.log2(self.edge_width / 4 + 1)
        color = self._edge_pen_color()
        if not self.edge_curves:
            for w, f, t in zip(widths, srcs, dests):
                edge = pg.PlotCurveItem(
                    [x[f], x[t]], [y[f], y[t]],
                    pen=pg.mkPen(color, width=w, cosmetic=True),
                    antialias=True)
                self.plot_widget.addItem(edge)
                self.edge_curves.append(edge)
            self._put_nodes_on_top()
        else:
            for edge, w, f, t in zip(self.edge_curves, widths, srcs, dests):
                edge.setData(
                    [x[f], x[t]], [y[f], y[t]],
                    pen=pg.mkPen(color, width=w, cosmetic=True),
                    antialias=True)

    def _put_nodes_on_top(self):
        if self.scatterplot_item:
            self.plot_widget.removeItem(self.scatterplot_item_sel)
            self.plot_widget.removeItem(self.scatterplot_item)
            self.plot_widget.addItem(self.scatterplot_item_sel)
            self.plot_widget.addItem(self.scatterplot_item)

    def set_edge_pen(self):
        if self.edge_curve:
            self.edge_curve.setPen(self._edge_curve_pen())
        elif self.edge_curves:
            self.update_edges_relative()

    def _edge_pen_color(self):
        return 0.5 if self.class_density else 0.8

    def _edge_curve_pen(self):
        return pg.mkPen(
            self._edge_pen_color(), width=self.edge_width, cosmetic=True)

    def set_edge_labels(self):
        pass

    def _remove_edges(self):
        if self.edge_curves:
            for edge in self.edge_curves:
                self.plot_widget.removeItem(edge)
            self.edge_curves.clear()
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
