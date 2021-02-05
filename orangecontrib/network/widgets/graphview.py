import time

import numpy as np
import pyqtgraph as pg

from AnyQt.QtCore import QLineF
from AnyQt.QtGui import QPen

from Orange.util import scale
from Orange.widgets.settings import Setting
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase


class PlotVarWidthCurveItem(pg.PlotCurveItem):
    def __init__(self, directed, *args, **kwargs):
        self.directed = directed
        self.widths = kwargs.pop("widths", None)
        self.pen = kwargs.pop("pen", pg.mkPen(0.0))
        self.setPen(self.pen)
        self.sizes = kwargs.pop("size", None)
        self.coss = self.sins = None
        super().__init__(*args, **kwargs)

    def setWidths(self, widths):
        self.widths = widths
        self.update()

    def setPen(self, pen):
        self.pen = pen

    def setData(self, *args, **kwargs):
        self.widths = kwargs.pop("widths", self.widths)
        self.pen = kwargs.pop("pen", self.pen)
        self.sizes = kwargs.pop("size", self.sizes)
        super().setData(*args, **kwargs)

    def paint(self, p, opt, widget):
        def get_arrows():
            cos12 = 10 * np.cos(np.pi / 12)
            sin12 = 10 * np.sin(np.pi / 12)

            # cos(a ± 12) = cos(a) cos(12) ∓ sin(a) sin(12)
            tx = sins * (fx * sin12)
            xa1s = x1s - coss * (fx * cos12)
            xa2s = xa1s - tx
            xa1s += tx

            # sin(a ± 12) = sin(a) cos(12) ± sin(12) cos(a)
            ty = (fy * sin12) * coss
            ya1s = y1s + sins * (fy * cos12)
            ya2s = ya1s - ty
            ya1s += ty
            return xa1s, ya1s, xa2s, ya2s

        def get_angles():
            angles = np.arctan2(-(y1s - y0s) / fy, (x1s - x0s) / fx)
            return np.cos(angles), np.sin(angles)
            """
            # This below faster. Uncomment and check that it works
            diffx, diffy = (x1s - x0s) / fx, (y1s - y0s) / fy
            norm = np.sqrt(diffx ** 2 + diffy ** 2)
            self.coss = np.nan_to_num(diffx / norm)
            self.sins = np.nan_to_num(diffy / norm)
            """

        def shorter_edges():
            nonlocal x0s, x1s, y0s, y1s
            sizes0, sizes1 = self.sizes[::2], self.sizes[1::2]
            return (x0s + fx * sizes0 * coss, y0s - fy * sizes0 * sins,
                    x1s - fx * sizes1 * coss, y1s + fy * sizes1 * sins)

        if self.xData is None or len(self.xData) == 0:
            return
        x0s, x1s = self.xData[::2], self.xData[1::2]
        y0s, y1s = self.yData[::2], self.yData[1::2]
        fx = 1 / p.worldTransform().m11()
        fy = 1 / p.worldTransform().m22()
        coss, sins = get_angles()
        endpoints = x0s, y0s, x1s, y1s = shorter_edges()

        p.setRenderHint(p.Antialiasing, True)
        p.setCompositionMode(p.CompositionMode_SourceOver)
        if self.widths is None:
            p.setPen(self.pen)
            if self.directed:
                for x0, y0, x1, y1, xa1, ya1, xa2, ya2 in zip(
                        *endpoints, *get_arrows()):
                    p.drawLine(QLineF(x0, y0, x1, y1))
                    p.drawLine(QLineF(xa1, ya1, x1, y1))
                    p.drawLine(QLineF(xa2, ya2, x1, y1))
            else:
                for x0, y0, x1, y1 in zip(*endpoints):
                    p.drawLine(QLineF(x0, y0, x1, y1))
        else:
            pen = QPen(self.pen)
            if self.directed:
                for x0, y0, x1, y1, xa1, ya1, xa2, ya2, w in zip(
                        *endpoints, *get_arrows(), self.widths):
                    pen.setWidth(w)
                    p.setPen(pen)
                    p.drawLine(QLineF(x0, y0, x1, y1))
                    p.drawLine(QLineF(xa1, ya1, x1, y1))
                    p.drawLine(QLineF(xa2, ya2, x1, y1))
            else:
                for x0, y0, x1, y1, w in zip(*endpoints, self.widths):
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
        self.step_resizing.connect(self.update_edges)
        self.end_resizing.connect(self.update_edges)

    def clear(self):
        super().clear()
        self._reset_attributes()

    def _reset_attributes(self):
        self.pair_indices = None
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
            self.pair_indices = np.empty((2 * len(srcs),), dtype=int)
            self.pair_indices[::2] = srcs
            self.pair_indices[1::2] = dests

        data = dict(x=x[self.pair_indices], y=y[self.pair_indices],
                    pen=self._edge_curve_pen(), antialias=True,
                    size=self.scatterplot_item.data["size"][self.pair_indices] / 2)
        if self.relative_edge_widths and len(set(weights)) > 1:
            data['widths'] = \
                scale(weights, .7, 8) * np.log2(self.edge_width / 4 + 1)
        else:
            data['widths'] = None

        if self.edge_curve is None:
            self.edge_curve = PlotVarWidthCurveItem(
                self.master.is_directed(), **data)
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
            0.5 if self.class_density else 0.3,
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
            num_selected = np.sum(selected)
            if num_selected >= 2:
                selected_edges = selected[srcs] & selected[dests]
            else:
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
            selection = np.array(self.selection, dtype=bool)
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

        self.update_edge_labels()
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
        self.last_click = (time.time(), indices)
        if time.time() - last_time < 0.5 and indices == last_indices:
            indices = self.master.get_reachable(indices)
        self.select_by_indices(indices)

    def unselect_all(self):
        super().unselect_all()
        if self.label_selected_edges:
            self.update_edge_labels()

    def _update_after_selection(self):
        if self.label_selected_edges:
            self.update_edge_labels()
        super()._update_after_selection()
