
from functools import partial
from itertools import zip_longest, chain

import numpy as np
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Orange import data
from Orange.util import scale
from Orange.widgets import gui
from Orange.widgets.utils.colorpalette import ColorPaletteGenerator, GradientPaletteGenerator
from Orange.projection.manifold import MDS

import networkx as nx
import pyqtgraph as pg

#~ class NodeItem(orangeqt.NodeItem):
    #~ def __init__(self, index, x=None, y=None, parent=None):
        #~ orangeqt.NodeItem.__init__(self, index, OWPoint.Ellipse, Qt.blue, 5, parent)
        #~ if x is not None:
            #~ self.set_x(x)
        #~ if y is not None:
            #~ self.set_y(y)
#~
#~ class EdgeItem(orangeqt.EdgeItem):
    #~ def __init__(self, u=None, v=None, weight=1, links_index=0, arrows=None, label='', parent=None):
        #~ orangeqt.EdgeItem.__init__(self, u, v, parent)
        #~ self.set_weight(weight)
        #~ self.set_links_index(links_index)
        #~ if arrows is not None:
            #~ self.set_arrows(arrows)


CONTINUOUS_PALETTE = GradientPaletteGenerator('#0000ff', '#ff0000')

class NodePenColor:
    DEFAULT = .3
    SELECTED = '#dd1100'
    HIGHLIGHTED = '#dd7700'

class NodePen:
    DEFAULT = pg.mkPen(NodePenColor.DEFAULT, width=1)
    SELECTED = pg.mkPen(NodePenColor.SELECTED, width=3)
    HIGHLIGHTED = pg.mkPen(NodePenColor.HIGHLIGHTED, width=3)

DEFAULT_EDGE_PEN = pg.mkPen('#ccc')

ANIMATION_ITERATIONS = 50


class NetworkCurve:
    def __init__(self, parent=None, pen=QPen(Qt.black), xData=None, yData=None):
        self.name = "Network Curve"

    def layout_fr(self, steps, weighted=False, smooth_cooling=False):
        orangeqt.NetworkCurve.fr(self, steps, weighted, smooth_cooling)

    def set_node_sizes(self, values={}, min_size=0, max_size=0):
        orangeqt.NetworkCurve.set_node_sizes(self, values, min_size, max_size)

    def fragviz_callback(self, a, b, mds, mdsRefresh, graph_components, matrix_components, progress_callback):
        """Refresh the UI when running  MDS on network components."""

        if not self.mdsStep % mdsRefresh:
            rotationOnly = False
            component_props = []
            x_mds = []
            y_mds = []
            phi = [None] * len(graph_components)
            nodes = self.nodes()
            ncomponents = len(graph_components)

            for i in range(ncomponents):

                if len(mds.points) == ncomponents:  # if average linkage before
                    x_avg_mds = mds.points[i][0]
                    y_avg_mds = mds.points[i][1]
                else:                                   # if not average linkage before
                    x = [mds.points[u][0] for u in matrix_components[i]]
                    y = [mds.points[u][1] for u in matrix_components[i]]

                    x_avg_mds = sum(x) / len(x)
                    y_avg_mds = sum(y) / len(y)
                    # compute rotation angle
#                    c = [np.linalg.norm(np.cross(mds.points[u], \
#                                [nodes[u].x(), nodes[u].y()])) for u in component]
#
#                    n = [np.vdot([nodes[u].x(), nodes[u].y()], \
#                                    [nodes[u].x(), nodes[u].y()]) for u in component]
#                    phi[i] = sum(c) / sum(n)


                x = [nodes[j].x() for j in graph_components[i]]
                y = [nodes[j].y() for j in graph_components[i]]

                x_avg_graph = sum(x) / len(x)
                y_avg_graph = sum(y) / len(y)

                x_mds.append(x_avg_mds)
                y_mds.append(y_avg_mds)

                component_props.append((x_avg_graph, y_avg_graph, \
                                        x_avg_mds, y_avg_mds, phi))

            for i, component in enumerate(graph_components):
                x_avg_graph, y_avg_graph, x_avg_mds, \
                y_avg_mds, phi = component_props[i]

    #            if phi[i]:  # rotate vertices
    #                #print "rotate", i, phi[i]
    #                r = np.array([[np.cos(phi[i]), -np.sin(phi[i])], [np.sin(phi[i]), np.cos(phi[i])]])  #rotation matrix
    #                c = [x_avg_graph, y_avg_graph]  # center of mass in FR coordinate system
    #                v = [np.dot(np.array([self.graph.coors[0][u], self.graph.coors[1][u]]) - c, r) + c for u in component]
    #                self.graph.coors[0][component] = [u[0] for u in v]
    #                self.graph.coors[1][component] = [u[1] for u in v]

                # translate vertices
                if not rotationOnly:
                    self.set_node_coordinates(dict(
                       (j, ((nodes[j].x() - x_avg_graph) + x_avg_mds,
                            (nodes[j].y() - y_avg_graph) + y_avg_mds)) \
                                  for j in component))

            #if self.mdsType == MdsType.exactSimulation:
            #    self.mds.points = [[self.graph.coors[0][i], \
            #                        self.graph.coors[1][i]] \
            #                        for i in range(len(self.graph.coors))]
            #    self.mds.freshD = 0

            self.plot().update_graph_layout()
            qApp.processEvents()

            if progress_callback is not None:
                progress_callback(a, self.mdsStep)

        self.mdsStep += 1
        return 0 if self.stopMDS else 1

    def layout_fragviz(self, steps, distances, graph, progress_callback=None, opt_from_curr=False):
        """Position the network components according to similarities among
        them.

        """

        if distances == None or graph == None or distances.dim != graph.number_of_nodes():
            self.information('invalid or no distance matrix')
            return 1

        p = self.plot()
        edges = self.edges()
        nodes = self.nodes()

        avgLinkage = True
        rotationOnly = False
        minStressDelta = 0
        mdsRefresh = 10#int(steps / 20)

        self.mdsStep = 1
        self.stopMDS = False

        nodes_inds = dict((n, i) for i, n in enumerate(sorted(graph.nodes_iter())))
        inds_nodes = dict((i, n) for i, n in enumerate(sorted(graph.nodes_iter())))

        graph_components = nx.algorithms.connected_components(graph)
        matrix_components = [[nodes_inds[n] for n in c] for c in graph_components]

        #~ distances.matrixType = core.SymMatrix.Symmetric

        # scale net coordinates
        if avgLinkage:
            distances = distances.avgLinkage(matrix_components)

        # if only one component
        if distances.dim == 1:
            return 0

        mds = MDS()(distances)
        rect = self.data_rect()
        w_fr = rect.width()
        h_fr = rect.height()
        d_fr = math.sqrt(w_fr ** 2 + h_fr ** 2)

        x_mds, y_mds = zip(*mds.points)
        w_mds = max(x_mds) - min(x_mds)
        h_mds = max(y_mds) - min(y_mds)
        d_mds = math.sqrt(w_mds ** 2 + h_mds ** 2)

        # if only one component
        if d_mds == 0 or d_fr == 0:
            d_mds = 1
            d_fr = 1

        self.set_node_coordinates(dict((key, (node.x() * d_mds / d_fr, node.y() * d_mds / d_fr)) \
                                       for key, node in nodes.items()))

        p.update_graph_layout()
        qApp.processEvents()

        if opt_from_curr:
            if avgLinkage:
                for u, c in enumerate(graph_components):
                    x = sum([nodes[n].x() for n in c]) / len(c)
                    y = sum([nodes[n].y() for n in c]) / len(c)
                    mds.points[u][0] = x
                    mds.points[u][1] = y
            else:
                for i, u in enumerate(sorted(nodes.keys())):
                    mds.points[i][0] = nodes[u].x()
                    mds.points[i][1] = nodes[u].y()
        else:
            mds.Torgerson()

        mds.optimize(steps, projection.mds.SgnRelStress, minStressDelta,
                     progressCallback=
                         lambda a, b=None, mds=mds, mdsRefresh=mdsRefresh, graph_comp=graph_components,
                                matrix_comp=matrix_components, progress_callback=progress_callback:
                         self.fragviz_callback(a, b, mds, mdsRefresh, graph_comp, matrix_comp, progress_callback))

        self.fragviz_callback(mds.avgStress, 0, mds, mdsRefresh, graph_components, matrix_components, progress_callback)

        if progress_callback is not None:
            progress_callback(mds.avgStress, self.mdsStep)

        return 0

    def mds_callback(self, a, b, mds, mdsRefresh, progress_callback):
        """Refresh the UI when running  MDS."""

        if not self.mdsStep % mdsRefresh:

            self.set_node_coordinates(dict((n, (mds.points[i][0], \
                                           mds.points[i][1])) for i, n in enumerate(sorted(self.nodes()))))
            self.plot().update_graph_layout()
            qApp.processEvents()

            if progress_callback is not None:
                progress_callback(a, self.mdsStep)

        self.mdsStep += 1
        return 0 if self.stopMDS else 1

    def layout_mds(self, steps, distances, progress_callback=None, opt_from_curr=False):
        """Position the network components according to similarities among
        them.

        """
        nodes = self.nodes()

        if distances == None or distances.dim != len(nodes):
            self.information('invalid or no distance matrix')
            return 1

        p = self.plot()

        minStressDelta = 0
        mdsRefresh = int(steps / 20)

        self.mdsStep = 1
        self.stopMDS = False

        #~ distances.matrixType = core.SymMatrix.Symmetric
        mds = MDS()(distances)
        mds.optimize(10, projection.mds.SgnRelStress, 0)
        rect = self.data_rect()
        w_fr = rect.width()
        h_fr = rect.height()
        d_fr = math.sqrt(w_fr ** 2 + h_fr ** 2)

        x_mds, y_mds = zip(*mds.points)
        w_mds = max(x_mds) - min(x_mds)
        h_mds = max(y_mds) - min(y_mds)
        d_mds = math.sqrt(w_mds ** 2 + h_mds ** 2)

        self.set_node_coordinates(dict(
           (n, (nodes[n].x() * d_mds / d_fr, nodes[n].y() * d_mds / d_fr)) for n in nodes))

        p.update_graph_layout()
        qApp.processEvents()

        if opt_from_curr:
            for i, u in enumerate(sorted(nodes.keys())):
                mds.points[i][0] = nodes[u].x()
                mds.points[i][1] = nodes[u].y()
        else:
            mds.Torgerson()

        mds.optimize(steps, projection.mds.SgnRelStress, minStressDelta,
                     progressCallback=
                         lambda a,
                                b=None,
                                mds=mds,
                                mdsRefresh=mdsRefresh,
                                progress_callback=progress_callback:
                                    self.mds_callback(a, b, mds, mdsRefresh, progress_callback))

        self.mds_callback(mds.avgStress, 0, mds, mdsRefresh, progress_callback)

        if progress_callback is not None:
            progress_callback(mds.avgStress, self.mdsStep)

        return 0

#    def move_selected_nodes(self, dx, dy):
#        selected = self.get_selected_nodes()
#
#        self.coors[selected][0] = self.coors[0][selected] + dx
#        self.coors[1][selected][1] = self.coors[1][selected] + dy
#
#        self.update_properties()
#        return selected
#
#    def set_hidden_nodes(self, nodes):
#        for vertex in self.nodes().itervalues():
#            vertex.setVisible(vertex.index() in nodes)
#
#    def hide_selected_nodes(self):
#        for vertex in self.nodes().itervalues():
#          if vertex.selected:
#            vertex.hide()
#
#    def hide_unselected_nodes(self):
#        for vertex in self.nodes().itervalues():
#          if not vertex.selected:
#            vertex.hide()
#
#    def show_all_vertices(self):
#        for vertex in self.nodes().itervalues():
#          vertex.show()


class OWNxCanvas(pg.GraphItem):
    def __init__(self, parent=None, name="Net Explorer"):
        super().__init__()
        self.setParent(parent)
        self.widget = parent

        self.kwargs = {'pen': DEFAULT_EDGE_PEN}
        self.textItems = []
        self.edgeTextItems = []
        self.graph = None
        self._pos_dict = None
        self.layout_fhr(False)
        self.show_edge_weights = False
        self.relative_edge_widths = False
        self.edgeColors = []
        self.selectedNodes = set()
        self.highlightedNodes = set()
        self.networkCurve = NetworkCurve()
        self.scatter.sigClicked.connect(self.pointClicked)

    def set_hidden_nodes(self, nodes):
        self.networkCurve.set_hidden_nodes(nodes)

    def hide_selected_nodes(self):
        self.networkCurve.hide_selected_nodes()
        self.drawPlotItems()

    def hide_unselected_nodes(self):
        self.networkCurve.hide_unselected_nodes()
        self.drawPlotItems()

    def show_all_vertices(self):
        self.networkCurve.show_all_vertices()
        self.drawPlotItems()

    def selected_nodes(self):
        return [vertex.index() for vertex in self.networkCurve.nodes().values() if vertex.is_selected()]
        #return [p.index() for p in self.selected_points()]

    def not_selected_nodes(self):
        return [vertex.index() for vertex in self.networkCurve.nodes().values() if not vertex.is_selected()]

    def marked_nodes(self):
        return [vertex.index() for vertex in self.networkCurve.nodes().values() if vertex.is_marked()]
        #return [p.index() for p in self.marked_points()]

    def not_marked_nodes(self):
        return [vertex.index() for vertex in self.networkCurve.nodes().values() if not vertex.is_marked()]

    def get_neighbors_upto(self, ndx, dist):
        newNeighbours = neighbours = set([ndx])
        for d in range(dist):
            tNewNeighbours = set()
            for v in newNeighbours:
                tNewNeighbours |= set(self.graph.neighbors(v))
            newNeighbours = tNewNeighbours - neighbours
            neighbours |= newNeighbours
        return neighbours

    def drawComponentKeywords(self):
        self.clear_markers()
        if not hasattr(self, "showComponentAttribute") or self.showComponentAttribute is None or self.graph is None or self.items is None:
            return

        if str(self.showComponentAttribute) not in self.items.domain:
            self.showComponentAttribute = None
            return

        components = nx.algorithms.components.connected_components(self.graph)
        nodes = self.networkCurve.nodes()

        for c in components:
            if len(c) == 0:
                continue

            x1 = sum(nodes[n].x() for n in c) / len(c)
            y1 = sum(nodes[n].y() for n in c) / len(c)
            lbl = str(self.items[c[0]][str(self.showComponentAttribute)])

            self.add_marker(lbl, x1, y1, alignment=Qt.AlignCenter, size=self.fontSize)

    def getColorIndeces(self, table, attribute, palette):
        colorIndices = {}
        colorIndex = None
        minValue = None
        maxValue = None

        if attribute[0] != "(" or attribute[ -1] != ")":
            i = 0
            for var in table.domain.variables:
                if var.name == attribute:
                    colorIndex = i
                    if var.varType == core.VarTypes.Discrete:
                        colorIndices = getVariableValueIndices(var, colorIndex)

                i += 1
            metas = table.domain.getmetas()
            for i, var in metas.items():
                if var.name == attribute:
                    colorIndex = i
                    if var.varType == core.VarTypes.Discrete:
                        colorIndices = getVariableValueIndices(var, colorIndex)

        colorIndices['?'] = len(colorIndices)
        palette.setNumberOfColors(len(colorIndices))

        if colorIndex != None and table.domain[colorIndex].varType == core.VarTypes.Continuous:
            minValue = float(min([x[colorIndex].value for x in table if x[colorIndex].value != "?"] or [0.0]))
            maxValue = float(max([x[colorIndex].value for x in table if x[colorIndex].value != "?"] or [0.0]))

        return colorIndices, colorIndex, minValue, maxValue

    def set_node_colors(self, attribute=None, replot=True):
        assert not attribute or isinstance(attribute, data.Variable)

        if not attribute:
            self.kwargs.pop('brush', None)
            if replot:
                self.replot()
            return

        table = self.graph.items()
        if not table:
            return
        if table.domain.class_var == attribute:
            values = table[:, attribute].Y
        else:
            values = table[:, attribute].X[:, 0]
        if attribute.is_continuous:
            colors = CONTINUOUS_PALETTE[scale(values)]
        elif attribute.is_discrete:
            DISCRETE_PALETTE = ColorPaletteGenerator(len(attribute.values))
            colors = DISCRETE_PALETTE[values]
        brushes = [QBrush(qcolor) for qcolor in colors]
        self.kwargs['brush'] = brushes
        if replot:
            self.replot()

    def set_node_labels(self, attributes=[]):
        assert isinstance(attributes, list)
        if not self.graph:
            for i in self.textItems: i.setText('')
            return
        if attributes:
            table = self.graph.items()
            for i, item in enumerate(self.textItems):
                text = ', '.join(map(str, table[i, attributes][0].list))
                item.setText(text, (30, 30, 30))
        else:
            for item in self.textItems:
                item.setText('')
        self.replot()

    def set_node_sizes(self, attribute=None, max_size=None, invert=False, replot=True):
        MIN_SIZE = 5
        try:
            table = self.graph.items()
            if attribute.is_string:
                values = np.array([i.list[0].count(',') + 1
                                   for i in table[:, attribute]])
            else:
                values = np.array(table[:, attribute]).T[0]
        except (AttributeError, TypeError, KeyError):
            self.kwargs['size'] = MIN_SIZE
        else:
            if invert:
                values += np.nanmin(values) + 1
                values = 1/values
            k, n = np.polyfit([np.nanmin(values), np.nanmax(values)], [MIN_SIZE, max_size], 1)
            sizes = values * k + n
            sizes[np.isnan(sizes)] = np.nanmean(sizes)
            self.kwargs['size'] = sizes
        finally:
            if replot:
                self.replot()

    def set_edge_sizes(self, replot=True):
        self.kwargs['pen'] = DEFAULT_EDGE_PEN
        G = self.graph
        if self.relative_edge_widths or self.edgeColors:
            if self.relative_edge_widths:
                weights = [G.edge[u][v].get('weight', 1) for u, v in G.edges()]
            else:
                weights = [1] * G.number_of_edges()
            if self.edgeColors:
                colors = self.edgeColors
            else:
                colors = [(0xb4, 0xb4, 0xb4)] * G.number_of_edges()

            self.kwargs['pen'] = np.array([c + (0xff, w) for c, w in zip(colors, weights)],
                                          dtype=[('red',   int),
                                                 ('green', int),
                                                 ('blue',  int),
                                                 ('alpha', int),
                                                 ('width', float)])
            self.kwargs['pen']['width'] = 5 * scale(self.kwargs['pen']['width'], .1, 1)
        if replot:
            self.replot()

    def set_edge_colors(self, attribute):
        colors = []
        if attribute:
            assert self.graph.links()
            values = self.graph.links()[:, attribute].X[:, 0]
            if attribute.is_continuous:
                colors = (tuple(i) for i in CONTINUOUS_PALETTE.getRGB(scale(values)))
            elif attribute.is_discrete:
                DISCRETE_PALETTE = ColorPaletteGenerator(len(attribute.values))
                colors = (DISCRETE_PALETTE[i] for i in values)
                colors = (tuple(c.red(), c.green(), c.blue()) for c in colors)
        self.edgeColors = colors
        self.set_edge_sizes()

    def set_edge_labels(self, attributes=[]):
        G = self.graph
        if not G: return
        weights = ''
        if self.show_edge_weights:
            weights = ['{:.2f}'.format(G.edge[u][v].get('weight', 0))
                       for u, v in G.edges()]

        for i in self.edgeTextItems:
            i.scene().removeItem(i)
        self.edgeTextItems = []
        try: table = G.links()[:, attributes]
        except TypeError: table = []
        for weight, row in zip_longest(weights, table):
            text = ','.join(chain([weight] if weight else [],
                                  row[0].list if row else []))
            item = pg.TextItem(text, (150, 150, 150))
            item.setParentItem(self)
            self.edgeTextItems.append(item)
        self.replot()

    def set_tooltip_attributes(self, attributes=[]):
        assert isinstance(attributes, list)
        self.tooltip_attributes = attributes

    def change_graph(self, newgraph):
        old_nodes = set(self.graph.nodes_iter())
        new_nodes = set(newgraph.nodes_iter())
        inter_nodes = old_nodes & new_nodes
        remove_nodes = list(old_nodes - inter_nodes)
        add_nodes = list(new_nodes - inter_nodes)

        self.graph = newgraph

        if len(remove_nodes) == 0 and len(add_nodes) == 0:
            return False

        current_nodes = self.networkCurve.nodes()

        center_x = np.average([node.x() for node in current_nodes.values()]) if len(current_nodes) > 0 else 0
        center_y = np.average([node.y() for node in current_nodes.values()]) if len(current_nodes) > 0 else 0

        def closest_nodes_with_pos(nodes):

            neighbors = set()
            for n in nodes:
                neighbors |= set(self.graph.neighbors(n))

            # checked all, none found
            if len(neighbors - nodes) == 0:
                return []

            inter = old_nodes.intersection(neighbors)
            if len(inter) > 0:
                return inter
            else:
                return closest_nodes_with_pos(neighbors | nodes)

        pos = dict((n, [np.average(c) for c in zip(*[(current_nodes[u].x(), current_nodes[u].y()) for u in closest_nodes_with_pos(set([n]))])]) for n in add_nodes)

        self.networkCurve.remove_nodes(remove_nodes)

        nodes = dict((v, self.NodeItem(v, x=pos[v][0] if len(pos[v]) == 2 else center_x, y=pos[v][1] if len(pos[v]) == 2 else center_y, parent=self.networkCurve)) for v in add_nodes)
        self.networkCurve.add_nodes(nodes)
        nodes = self.networkCurve.nodes()

        #add edges
        new_edges = self.graph.edges(add_nodes)

        if self.links is not None and len(self.links) > 0:
            links = self.links
            links_indices = (self.edge_to_row[i + 1][j + 1] for (i, j) in new_edges)

            if self.graph.is_directed():
                edges = [EdgeItem(nodes[i], nodes[j],
                    self.graph[i][j].get('weight', 1), links_index, arrows=EdgeItem.ArrowV, \
                    parent=self.networkCurve) for ((i, j), links_index) in \
                         zip(new_edges, links_indices)]
            else:
                edges = [EdgeItem(nodes[i], nodes[j],
                    self.graph[i][j].get('weight', 1), links_index) for \
                    ((i, j), links_index) in zip(new_edges, \
                                        links_indices, parent=self.networkCurve)]
        elif self.graph.is_directed():
            edges = [EdgeItem(nodes[i], nodes[j], self.graph[i][j].get('weight', 1), \
                    arrows=EdgeItem.ArrowV, parent=self.networkCurve) for (i, j) in new_edges]
        else:
            edges = [EdgeItem(nodes[i], nodes[j], self.graph[i][j].get('weight', 1), \
                    parent=self.networkCurve) for (i, j) in new_edges]

        self.networkCurve.add_edges(edges)

        if len(current_nodes) < 3:
            self.networkCurve.random()

        return True

    def set_antialias(self, value):
        self.kwargs['antialias'] = bool(value)

    def set_graph(self, graph):
        self.graph = graph
        try:
            for i in self.textItems:
                i.scene().removeItem(i)
        except AttributeError: pass  # No scene
        if graph:
            # Adjacency matrix
            self.kwargs['adj'] = np.array(graph.edges())
            # Custom node data (index used in mouseDragEvent)
            self.kwargs['data'] = np.arange(graph.number_of_nodes())
            # Construct empty node labels
            self.textItems = []
            for i in range(graph.number_of_nodes()):
                item = pg.TextItem()
                self.textItems.append(item)
                item.setParentItem(self)
            # Node pens
            pens = [NodePen.DEFAULT] * graph.number_of_nodes()
            for i in self.selectedNodes:
                pens[i] = NodePen.SELECTED
            self.kwargs['symbolPen'] = pens
            # Set edge weights
            self.set_edge_sizes(replot=False)
            self.set_node_colors(replot=False)
            self.set_node_sizes(replot=False)
            # Position nodes according to selected layout optimization
            self.layout_func()
        self.replot()

    def set_labels_on_marked(self, labelsOnMarkedOnly):
        self.networkCurve.set_labels_on_marked(labelsOnMarkedOnly)
        self.set_node_labels()
        self.replot()

    def update_graph_layout(self):
        self._bounds_cache = {}
        self._transform_cache = {}
        ...

    def layout_original(self):
        def _f():
            items = self.graph.items()
            if not items or 'x' not in items.domain or 'y' not in items.domain:
                return nx.spring_layout(self.graph,
                                        iterations=2,
                                        k=.5/np.sqrt(self.graph.number_of_nodes()))
            positions = {node: (items[node]['x'].value if items[node]['x'].value != '?' else np.random.rand(),
                                items[node]['y'].value if items[node]['y'].value != '?' else np.random.rand())
                         for node in self.graph.node}
            return positions
        return self._layout(lambda _: _f())

    _is_animating = False

    @property
    def is_animating(self):
        return self._is_animating

    @is_animating.setter
    def is_animating(self, value):
        self._is_animating = value
        if value:
            self.parent().setCursor(QCursor(Qt.ForbiddenCursor))
        else:
            self.parent().unsetCursor()
        qApp.processEvents()

    def _animate(self, newpos):
        if not self.graph: return
        self.is_animating = True
        n_iteration = ANIMATION_ITERATIONS
        pos = self.kwargs['pos']
        delta = (newpos - pos) / n_iteration
        if np.abs(delta).sum() > 1e-4:  # If not already there
            while self.is_animating and n_iteration:
                self.kwargs['pos'] += delta
                self.replot()
                self.animation_progress.advance()
                n_iteration -= 1
        self.is_animating = False

    def layout_fhr(self, weighted=False):
        def callback(pos):
            self.kwargs['pos'] = pos
            self.replot()
            self.animation_progress.advance()
            return self.is_animating

        from .._fr_layout import fruchterman_reingold_layout

        def run_fr(G):
            self.is_animating = True
            pos = self.kwargs.get('pos')
            if pos is not None and pos.shape[0] != G.number_of_nodes():
                pos = None
            return fruchterman_reingold_layout(G,
                                               iterations=ANIMATION_ITERATIONS,
                                               pos=pos,
                                               weight='weight' if weighted else None,
                                               callback=callback)
        self._layout(run_fr)

    def _layout(self, layout_func):
        def pos_array(pos):
            """Return ndarray of positions from node-to-position dict"""
            return np.array([pos[n] for n in sorted(self.graph)])
        def _relayout():
            if not self.graph: return []
            self.animation_progress = gui.ProgressBar(self.widget, ANIMATION_ITERATIONS)
            pos = pos_array(layout_func(self.graph))
            self._animate(pos)
            self.animation_progress.finish()
        self.layout_func = _relayout
        _relayout()

    def layout_circular(self):
        self._layout(nx.circular_layout)

    def layout_spectral(self):
        self._layout(partial(nx.spectral_layout))

    def layout_random(self):
        self._layout(nx.random_layout)

    def layout_concentric(self):
        def _generate_nlist():
            G = self.graph
            # TODO: imaginative, but shit. revise.
            isolates = set(nx.isolates(G))
            independent = set(nx.maximal_independent_set(G)) - isolates
            dominating = set(nx.dominating_set(G)) - independent - isolates
            rest = set(G.nodes()) - dominating - independent - isolates
            nlist = list(map(sorted, filter(None, (isolates, independent, dominating, rest))))
            return nlist
        nlist = _generate_nlist()
        self._layout(partial(nx.shell_layout, nlist=nlist))

    def replot(self):
        if not self.graph:
            self.adjacency = None  # bug in pyqtgraph.GraphItem
            super().setData(pos=[[.5, .5]])
            item = pg.TextItem('no network', 'k')
            self.textItems.append(item)
            item.setParentItem(self)
            item.setPos(.5, .5)
            return
        # Update scatter plot (graph)
        super().setData(**self.kwargs)
        # Update text labels
        for item, pos in zip(self.textItems, self.kwargs['pos']):
            item.setPos(*pos[:2])
        # Update text labels for edges
        pos = self.kwargs['pos']
        for item, adj in zip(self.edgeTextItems, self.kwargs['adj']):
            item.setPos(*((pos[adj[0]] + pos[adj[1]]) / 2))
        qApp.processEvents()

    def hoverEvent(self, ev):
        self.scatter.setToolTip('')
        try: point = self.scatter.pointsAt(ev.pos())
        except AttributeError: return  # HoverEvent() excepts if tooltip if hovered (pyqtgraph bug, methinks)
        if not point: return
        ind = point[0].data()
        try: row = self.graph.items()[ind, self.tooltip_attributes][0].list
        except (TypeError, AttributeError): return
        self.scatter.setToolTip('<br>'.join('<b>{.name}:</b> {}'.format(i[0], str(i[1]).replace('<', '&lt;'))
                                            for i in zip(self.tooltip_attributes, row)))

    def updateNodeCounters(self):
        self.parent().nSelected = len(self.selectedNodes)
        self.parent().nHighlighted = len(self.highlightedNodes)
        # Export selected nodes
        self.parent().commit()

    def selectHighlighted(self):
        self.selectedNodes |= self.highlightedNodes
        self._setNodesPen(select=self.highlightedNodes)
        self.highlightedNodes = set()
        self.updateNodeCounters()

    def setHighlighted(self, nodes):
        nodes = nodes if isinstance(nodes, set) else set(nodes)
        nodes -= self.selectedNodes
        self._setNodesPen(highlight=nodes, default=self.highlightedNodes)
        self.highlightedNodes = nodes
        self.updateNodeCounters()

    def _setNodesPen(self, select=[], highlight=[], default=[]):
        points = self.scatter.points()
        def setPen(nodes, pen):
            for i in nodes:
                self.kwargs['symbolPen'][i] = pen
                points[i]._data['pen'] = pen
                points[i]._data['sourceRect'] = None
            self.scatter.updateSpots()

        if select:
            setPen(select, NodePen.SELECTED)
        if default: setPen(default, NodePen.DEFAULT)
        if highlight: setPen(highlight, NodePen.HIGHLIGHTED)

    def selectNodesInRect(self, qrect):
        if not self.graph: return
        top, bottom, left, right = qrect.top(), qrect.bottom(), qrect.left(), qrect.right()
        assert top < bottom and left < right
        data = self.scatter.data
        AND_ = np.logical_and
        selected = AND_(AND_(left <= data['x'], data['x'] <= right),
                        AND_(top <= data['y'], data['y'] <= bottom)).nonzero()[0]
        selected = set(selected)
        selected -= self.selectedNodes
        self._setNodesPen(select=selected)
        self.selectedNodes |= selected
        self.updateNodeCounters()

    def pointClicked(self, plot, points):
        if not self.graph: return
        i = points[0].data()
        is_selected = i not in self.selectedNodes
        if is_selected:
            self._setNodesPen(select=[i])
            self.selectedNodes.add(i)
            self.highlightedNodes.discard(i)
        else:
            self._setNodesPen(default=[i])
            self.selectedNodes.remove(i)
        self.updateNodeCounters()
        self.parent().set_mark_mode()  # Re-evaluate highlighted nodes

    def mouseClickEvent(self, ev):
        """Mouse clicked on canvas => clear selection"""
        self._setNodesPen(default=self.selectedNodes | self.highlightedNodes)
        self.selectedNodes = set()
        self.highlightedNodes = set()
        self.updateNodeCounters()
        self.parent().set_mark_mode()

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            ind = self.dragPoint = pts[0].data()
            self.dragOffset = self.kwargs['pos'][ind][:2] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        ev.accept()
        ind = self.dragPoint
        change = ev.pos() - ev.lastPos()

        self.scatter.prepareGeometryChange()
        self.scatter.informViewBoundsChanged()
        self.scatter.bounds = [None, None]

        # If spot is part of the selection, also move the other selected spots
        to_move = self.selectedNodes if ind in self.selectedNodes else [ind]
        for i in to_move:
            self.kwargs['pos'][i][:2] += change
            item = self.scatter.data[i]['item']
            item._data['x'] += change[0]
            item._data['y'] += change[1]
            item.updateItem()

        self.generatePicture()
