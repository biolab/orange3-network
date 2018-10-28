import time

import numpy as np
import pyqtgraph as pg

from AnyQt.QtCore import QLineF
from AnyQt.QtGui import QPen

from Orange.util import scale
from Orange.widgets.settings import Setting
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase


class PlotVarWidthCurveItem(pg.PlotCurveItem):
    def __init__(self, *args, **kwargs):
        self.widths = kwargs.pop("widths", None)
        self.setPen(kwargs.pop("pen", pg.mkPen(0.0)))
        super().__init__(*args, **kwargs)

    def setWidths(self, widths):
        self.widths = widths
        self.update()

    def setPen(self, pen):
        self.pen = pen

    def setData(self, *args, **kwargs):
        self.widths = kwargs.pop("widths", self.widths)
        self.pen = kwargs.pop("pen", self.pen)
        super().setData(*args, **kwargs)

    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0:
            return
        p.setRenderHint(p.Antialiasing, True)
        p.setCompositionMode(p.CompositionMode_SourceOver)
        if self.widths is None:
            p.setPen(self.pen)
            for x0, y0, x1, y1 in zip(self.xData[::2], self.yData[::2],
                                      self.xData[1::2], self.yData[1::2]):
                p.drawLine(QLineF(x0, y0, x1, y1))
        else:
            pen = QPen(self.pen)
            for x0, y0, x1, y1, w in zip(self.xData[::2], self.yData[::2],
                                         self.xData[1::2], self.yData[1::2],
                                         self.widths):
                pen.setWidth(w)
                p.setPen(pen)
                p.drawLine(QLineF(x0, y0, x1, y1))


class GraphView(OWScatterPlotBase):
    show_edge_weights = Setting(False)
    relative_edge_widths = Setting(True)
    edge_width = Setting(2)
    label_selected_edges = Setting(True)

    COLOR_NOT_SUBSET = (255, 255, 255, 255)
    COLOR_SUBSET = (0, 0, 0, 255)
    COLOR_DEFAULT = (255, 255, 255, 0)

    class Simplifications:
        NoLabels, NoEdges, NoEdgeLabels, NoDensity, = 1, 2, 4, 8
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
        self.edge_labels = []
        self.scatterplot_marked = None
        self.last_click = (-1, None)

    def update_coordinates(self):
        super().update_coordinates()
        self.update_marks()
        self.update_edges()

    def set_simplifications(self, simplifications):
        S = self.Simplifications
        for flag, remove, update in (
                (S.NoDensity, self._remove_density, self.update_density),
                (S.NoLabels, self._remove_labels, self.update_labels),
                (S.NoEdges, self._remove_edges, self.update_edges),
                (S.NoEdgeLabels,
                 self._remove_edge_labels, self.update_edge_labels)):
            if simplifications & flag != self.simplify & flag:
                if simplifications & flag:
                    self.simplify += flag
                    remove()
                else:
                    self.simplify -= flag
                    update()

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

        data = dict(x=x[self.paired_indices], y=y[self.paired_indices],
                    pen=self._edge_curve_pen(), antialias=True)
        if self.relative_edge_widths and len(set(weights)) > 1:
            data['widths'] = \
                scale(weights, .7, 8) * np.log2(self.edge_width / 4 + 1)
        else:
            data['widths'] = None

        if self.edge_curve is None:
            self.edge_curve = PlotVarWidthCurveItem(**data)
            self.edge_curve.setZValue(-10)
            self.plot_widget.addItem(self.edge_curve)
        else:
            self.edge_curve.setData(**data)
        self.update_edge_labels()

    def set_edge_pen(self):
        if self.edge_curve:
            self.edge_curve.setPen(self._edge_curve_pen())

    def _edge_curve_pen(self):
        return pg.mkPen(
            0.5 if self.class_density else 0.8,
            width=self.edge_width,
            cosmetic=True)

    def update_edge_labels(self):
        for label in self.edge_labels:
            self.plot_widget.removeItem(label)
        self.edge_labels = []
        if self.scatterplot_item is None \
                or not self.show_edge_weights \
                or self.simplify & self.Simplifications.NoEdgeLabels:
            return
        edges = self.master.get_edges()
        if edges is None:
            return
        srcs, dests, weights = edges.row, edges.col, edges.data
        if self.label_selected_edges:
            selected = self._selected_and_marked()
            selected_edges = selected[srcs] | selected[dests]
            srcs = srcs[selected_edges]
            dests = dests[selected_edges]
            weights = weights[selected_edges]
        if np.allclose(weights, np.round(weights)):
            labels = [str(x) for x in weights.astype(np.int)]
        else:
            labels = ["{:.02}".format(x) for x in weights]
        x, y = self.scatterplot_item.getData()
        xs = (x[srcs.astype(np.int64)] + x[dests.astype(np.int64)]) / 2
        ys = (y[srcs.astype(np.int64)] + y[dests.astype(np.int64)]) / 2
        black = pg.mkColor(0, 0, 0)
        for label, x, y in zip(labels, xs, ys):
            ti = pg.TextItem(label, black)
            ti.setPos(x, y)
            self.plot_widget.addItem(ti)
            self.edge_labels.append(ti)

    def _remove_edges(self):
        if self.edge_curve:
            self.plot_widget.removeItem(self.edge_curve)
            self.edge_curve = None
        self._remove_edge_labels()

    def _remove_edge_labels(self):
        for label in self.edge_labels:
            self.plot_widget.removeItem(label)
        self.edge_labels = []

    def update_density(self):
        if not self.simplify & self.Simplifications.NoDensity:
            super().update_density()
            self.set_edge_pen()

    # pylint: disable=access-member-before-definition
    def _remove_density(self):
        if self.density_img:
            self.plot_widget.removeItem(self.density_img)
            self.density_img = None

    def _selected_and_marked(self):
        if self.selection is None:
            selection = np.zeros(len(self.scatterplot_item.data), dtype=bool)
        else:
            selection = np.array(self.selection, dtype=np.bool)
        marked = self.master.get_marked_nodes()
        if marked is not None:
            selection[marked] = 1
        return selection

    def update_labels(self):
        if self.simplify & self.Simplifications.NoLabels:
            return
        # This is not nice, but let's not add methods to the parent just
        # to support this specific needs of network explorer
        # pylint: disable=access-member-before-definition
        saved_selection = self.selection
        if self.label_only_selected and self.scatterplot_item:
            marked = self.master.get_marked_nodes()
            if marked is not None and len(marked):
                self.selection = self._selected_and_marked()
        super().update_labels()
        self.selection = saved_selection

    def _remove_labels(self):
        # pylint: disable=access-member-before-definition
        for label in self.labels:
            self.plot_widget.removeItem(label)
        self.labels = []

    def update_marks(self):
        if self.scatterplot_marked is None:
            self.scatterplot_marked = pg.ScatterPlotItem([], [])
            self.scatterplot_marked.setZValue(-5)
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
        if time.time() - last_time < 0.5 and indices == last_indices:
            indices = self.master.get_reachable(indices)
        self.last_click = (time.time(), indices)
        self.select_by_indices(indices)

    def unselect_all(self):
        super().unselect_all()
        if self.label_selected_edges:
            self.update_edge_labels()

    def _update_after_selection(self):
        if self.label_selected_edges:
            self.update_edge_labels()
        super()._update_after_selection()
