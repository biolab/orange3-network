
import numpy as np
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Orange import data
from Orange.util import scale
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

def pos_array(pos):
    """Return ndarray of positions from node-to-position dict"""
    return [i[1] for i in sorted(pos.items())]


CONTINUOUS_PALETTE = GradientPaletteGenerator('#0000ff', '#ff0000')


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
    def __init__(self, master, parent=None, name="None"):
        super().__init__()

        self.kwargs = {}
        self.textItems = []
        self.layout_fhr(False)

        self.master = master
        self.parent = parent
        self.graph = None

        self.circles = []
        self.freezeNeighbours = False
        self.labelsOnMarkedOnly = 0

        self.show_indices = False
        self.show_weights = False
        self.trim_label_words = 0
        self.show_component_distances = False

        self.showComponentAttribute = None
        self.forceVectors = None
        #self.appendToSelection = 1
        self.fontSize = 12

        self.networkCurve = NetworkCurve()

        self.minComponentEdgeWidth = 0
        self.maxComponentEdgeWidth = 0
        self.items_matrix = None

        self.items = None
        self.links = None
        self.label_distances = None

        self.node_label_attributes = []
        self.edge_label_attributes = []

        self.axis_margin = 0
        self.title_margin = 0
        self.graph_margin = 1
        self._legend_margin = QRectF(0, 0, 0, 0)

        #self.setFocusPolicy(Qt.StrongFocus)

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

    def mark_on_selection_changed(self):
        toMark = set()
        for ndx in self.selected_nodes():
            toMark |= self.get_neighbors_upto(ndx, self.mark_neighbors)

        self.networkCurve.clear_node_marks()
        self.networkCurve.set_node_marks(dict((i, True) for i in toMark))

    def mark_on_focus_changed(self, node):
        self.networkCurve.clear_node_marks()

        if node is not None:
            toMark = set(self.get_neighbors_upto(node.index(), self.mark_neighbors))
            self.networkCurve.set_node_marks(dict((i, True) for i in toMark))

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

    def set_node_colors(self, attribute=None):
        assert not attribute or isinstance(attribute, data.Variable)
        assert self.graph

        if not attribute:
            self.kwargs.pop('brush', None)
            self.replot()
            return

        table = self.graph.items()
        if not table:
            return
        values = table[:, attribute].X[:, 0]
        if attribute.is_continuous:
            colors = CONTINUOUS_PALETTE[scale(values)]
        elif attribute.is_discrete:
            DISCRETE_PALETTE = ColorPaletteGenerator(len(attribute.values))
            colors = (DISCRETE_PALETTE[i] for i in values)
        brushes = [QBrush(qcolor) for qcolor in colors]
        self.kwargs['brush'] = brushes
        self.replot()

    def set_node_labels(self, attributes=[]):
        assert isinstance(attributes, list)
        if attributes:
            table = self.graph.items()
            for i, item in enumerate(self.textItems):
                text = ', '.join(map(str, table[i, attributes][0].list))
                item.setText(text, (30, 30, 30))
        else:
            for item in self.textItems:
                item.setText('')
        self.replot()

    def set_node_sizes(self, attribute, max_size, invert):
        MIN_SIZE = 8
        try:
            table = self.graph.items()
            if attribute.is_string:
                values = np.array([len(i.list[0].split(',')) for i in table[:, attribute]])
            else:
                values = np.array(table[:, attribute]).T[0]
        except (AttributeError, TypeError, KeyError):
            self.kwargs['size'] = MIN_SIZE
        else:
            if invert:
                values = 1/values
            k, n = np.polyfit([np.nanmin(values), np.nanmax(values)], [MIN_SIZE, max_size], 1)
            sizes = values * k + n
            sizes[np.isnan(sizes)] = np.nanmean(sizes)
            self.kwargs['size'] = sizes
        finally:
            self.replot()

    def set_edge_colors(self, attribute):
        if self.graph is None:
            return

        colorIndices, colorIndex, minValue, maxValue = self.getColorIndeces(self.links, attribute, self.discPalette)
        colors = []

        if colorIndex is not None and self.links.domain[colorIndex].varType == core.VarTypes.Continuous and minValue == maxValue:
            colors = [self.discEdgePalette[0] for edge in self.networkCurve.edge_indices()]

        elif colorIndex is not None and self.links.domain[colorIndex].varType == core.VarTypes.Continuous:
            colors = [self.contPalette[(float(self.links[edge.links_index()][colorIndex].value) - minValue) / (maxValue - minValue)]
                          if str(self.links[edge.links_index()][colorIndex].value) != '?' else
                          self.discPalette[0] for edge in self.networkCurve.edges()]

        elif colorIndex is not None and self.links.domain[colorIndex].varType == core.VarTypes.Discrete:
            colors = [self.discEdgePalette[colorIndices[self.links[edge.links_index()][colorIndex].value]] for edge in self.networkCurve.edges()]

        else:
            colors = [self.discEdgePalette[0] for edge in self.networkCurve.edge_indices()]

        self.networkCurve.set_edge_colors(colors)
        self.replot()

    def set_edge_labels(self, attributes=None):
        if self.graph is None:
            return

        edges = self.networkCurve.edge_indices()

        if attributes is not None:
            self.edge_label_attributes = attributes

        label_attributes = []
        if self.links is not None and isinstance(self.links, data.Table):
            label_attributes = [self.links.domain[att] for att in \
                self.edge_label_attributes if att in self.links.domain]

        weights = [[] for ex in edges]
        if self.show_weights:
            weights = [["%.2f" % self.graph[u][v].get('weight', 1)] for u, v in edges]

        self.networkCurve.set_edge_labels([', '.join(weights[i] + \
                           [str(self.links[i][att]) for att in \
                           label_attributes]) for i, edge in enumerate(edges)])

        self.replot()

    def set_tooltip_attributes(self, attributes):
        if self.graph is None or self.items is None or \
           not isinstance(self.items, data.Table):
            return

        tooltip_attributes = [self.items.domain[att] for att in \
                                 attributes if att in self.items.domain]
        self.networkCurve.set_node_tooltips(dict((node, ', '.join(str( \
                   self.items[node][att]) for att in tooltip_attributes)) \
                                                        for node in self.graph))

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

    def set_graph(self, graph):
        self.graph = graph
        if graph:
            self.kwargs['adj'] = np.array(self.graph.edges())
            # Construct empty node labels
            self.textItems = []
            for i in range(graph.number_of_nodes()):
                item = pg.TextItem()
                self.textItems.append(item)
                item.setParentItem(self)
        self.replot()

    def set_labels_on_marked(self, labelsOnMarkedOnly):
        self.networkCurve.set_labels_on_marked(labelsOnMarkedOnly)
        self.set_node_labels()
        self.replot()

    def set_show_component_distances(self):
        self.networkCurve.set_show_component_distances(self.show_component_distances)
        self.replot()

    def update_graph_layout(self):
        self._bounds_cache = {}
        self._transform_cache = {}
        ...

    def layout_original(self):
        def _f(G):
            items = G.items()
            if not items or 'x' not in items.domain or 'y' not in items.domain:
                raise Exception('graph items table doesn\'t have x,y info')
            positions = {node: (items[node]['x'].value,
                                items[node]['y'].value)
                         for node in self.graph.node
                         if items[node]['x'].value != '?'
                         and items[node]['y'].value != '?'}
            if len(positions) == len(items):
                return pos_array(positions)
            else:
                return layout_fhr(pos=positions, iterations=1)(G)

    def layout_fhr(self, weighted=False, pos=None, iterations=50):
        def _f(G):
            return pos_array(nx.spring_layout(G, dim=3, pos=pos,
                                              iterations=iterations,
                                              weight='weight' if weighted else None))
        self.kwargs.pop('pos', None)
        self.layout_func = _f
        return self

    def layout_circular(self):
        def _f(G): return pos_array(nx.circular_layout(G))
        self.kwargs.pop('pos', None)
        self.layout_func = _f
        return self

    def layout_spectral(self):
        def _f(G): return pos_array(nx.spectral_layout(G, dim=3))
        self.kwargs.pop('pos', None)
        self.layout_func = _f
        return self

    def layout_random(self):
        def _f(G): return pos_array(nx.random_layout(G))
        self.kwargs.pop('pos', None)
        self.layout_func = _f
        return self

    def layout_concentric(self):
        def _f(G):
            isolates = set(nx.isolates(G))
            independent = set(nx.maximal_independent_set(G)) - isolates
            dominating = set(nx.dominating_set(G)) - independent - isolates
            rest = set(G.nodes()) - dominating - independent - isolates
            nlist = list(map(sorted, filter(None, (isolates, independent, dominating, rest))))
            return pos_array(nx.shell_layout(G, nlist=nlist))

        self.kwargs.pop('pos', None)
        self.layout_func = _f
        return self

    def _updateGraph(self):
        # Update scatter plot (graph)
        super().setData(**self.kwargs)
        # Update text labels
        for item, pos in zip(self.textItems, self.kwargs['pos']):
            item.setPos(*pos[:2])

    def replot(self):
        lines = []
        size = []
        data = []

        if not self.graph:
            self.setData(pos=[[.5, .5]], text=['no network'])
            return

        if 'pos' not in self.kwargs:
            self.kwargs['pos'] = np.array(self.layout_func(self.graph))

        self.kwargs['data'] = np.arange(self.graph.number_of_nodes())

        self._updateGraph()

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
            self.dragPoint = pts[0]
            ind = pts[0].data()
            self.dragOffset = self.kwargs['pos'][ind][:2] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()
        self.kwargs['pos'][ind][:2] = ev.pos() + self.dragOffset
        ev.accept()
        self._updateGraph()
