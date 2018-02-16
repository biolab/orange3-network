from threading import Thread

import numpy as np
import networkx as nx

from AnyQt import QtCore, QtGui
from AnyQt.QtCore import QLineF, QRectF, Qt
from AnyQt.QtGui import QBrush, QPen, QColor
from AnyQt.QtWidgets import qApp, QStyle, QGraphicsLineItem, QGraphicsEllipseItem, \
    QGraphicsView, QGraphicsScene, QWidget, QGraphicsSimpleTextItem

from orangecontrib.network._fr_layout import fruchterman_reingold_layout

from Orange.widgets.visualize.owdistributions import LegendItem as DistributionsLegendItem
# Expose OpenGL rendering for large graphs, if available
HAVE_OPENGL = True
try:
    from AnyQt import QtOpenGL
except:
    HAVE_OPENGL = False

FR_ITERATIONS = 250

IS_LARGE_GRAPH = lambda G: G.number_of_nodes() + G.number_of_edges() > 4000
IS_VERY_LARGE_GRAPH = lambda G: G.number_of_nodes() + G.number_of_edges() > 10000


class LegendItem(DistributionsLegendItem):
    def __init__(self):
        super().__init__()
        self.parent = None
        self.x = None
        self.y = None

    def set_parent(self, parent):
        self.parent = parent
        parent.scene().changed.connect(self.geometry_changed)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            event.accept()
            if self.parent is not None:
                new_pos = self.pos() + (event.pos() - event.lastPos())
                self.setPos(new_pos)
                new_pos = self.parent.mapFromScene(new_pos)
                self.x = new_pos.x()
                self.y = new_pos.y()
        else:
            event.ignore()

    def geometry_changed(self):
        a = self.parent.frameRect()
        x = a.width() - self.width()
        y = a.height() - self.height()
        if None in [self.x, self.y]:
            self.x = x
            self.y = 0
        inside = lambda x, min_x, max_x: max(min(max_x, x), min_x)
        self.x = inside(self.x, 0, x)
        self.y = inside(self.y, 0, y)
        self.setPos(self.parent.mapToScene(self.x, self.y))


class QGraphicsEdge(QGraphicsLineItem):
    def __init__(self, source, dest, view=None):
        super().__init__()
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setFlags(self.ItemIgnoresTransformations |
                      self.ItemIgnoresParentOpacity)
        self.setZValue(1)
        self.setPen(QPen(Qt.gray, .7))

        source.addEdge(self)
        dest.addEdge(self)
        self.source = source
        self.dest = dest
        self.__transform = view.transform
        # Add text labels
        label = self.label = QGraphicsSimpleTextItem('test', self)
        label.setVisible(False)
        label.setBrush(Qt.gray)
        label.setZValue(2)
        label.setFlags(self.ItemIgnoresParentOpacity |
                       self.ItemIgnoresTransformations)
        view.scene().addItem(label)

        self.adjust()

    def adjust(self):
        line = QLineF(self.mapFromItem(self.source, 0, 0),
                      self.mapFromItem(self.dest, 0, 0))
        self.label.setPos(line.pointAt(.5))
        self.setLine(self.__transform().map(line))


class Edge(QGraphicsEdge):
    def __init__(self, source, dest, view=None):
        super().__init__(source, dest, view)
        self.setSize(.7)

    def setSize(self, size):
        self.setPen(QPen(self.pen().color(), size))

    def setText(self, text):
        if text: self.label.setText(text)
        self.label.setVisible(bool(text))

    def setColor(self, color):
        self.setPen(QPen(QColor(color or Qt.gray), self.pen().width()))


class QGraphicsNode(QGraphicsEllipseItem):
    """This class is the bare minimum to sustain a connected graph"""
    def __init__(self, rect=QRectF(-5, -5, 10, 10), view=None):
        super().__init__(rect)
        self.setCacheMode(self.DeviceCoordinateCache)
        self.setAcceptHoverEvents(True)
        self.setFlags(self.ItemIsMovable |
                      self.ItemIsSelectable |
                      self.ItemIgnoresTransformations |
                      self.ItemIgnoresParentOpacity |
                      self.ItemSendsGeometryChanges)
        self.setZValue(4)

        self.edges = []
        self._radius = rect.width() / 2
        self.__transform = view.transform
        # Add text labels
        label = self.label = QGraphicsSimpleTextItem('test', self)
        label.setVisible(False)
        label.setFlags(self.ItemIgnoresParentOpacity |
                       self.ItemIgnoresTransformations)
        label.setZValue(3)
        view.scene().addItem(label)

    def setPos(self, x, y):
        self.adjust()
        super().setPos(x, y)

    def adjust(self):
        # Adjust label position
        d = 1 / self.__transform().m11() * self._radius
        self.label.setPos(self.pos().x() + d, self.pos().y() + d)

    def addEdge(self, edge):
        self.edges.append(edge)

    def itemChange(self, change, value):
        if change == self.ItemPositionHasChanged:
            self.adjust()
            for edge in self.edges:
                edge.adjust()
        return super().itemChange(change, value)


class Node(QGraphicsNode):
    """
    This class provides an interface for all the bells & whistles of the
    Network Explorer.
    """

    BRUSH_DEFAULT = QBrush(QColor('#669'))

    class Pen:
        DEFAULT = QPen(Qt.black, 0)
        SELECTED = QPen(QColor('#dd0000'), 3)
        HIGHLIGHTED = QPen(QColor('#ffaa22'), 3)

    _TOOLTIP = lambda: ''

    def __init__(self, id, view):
        super().__init__(view=view)
        self.id = id
        self.setBrush(Node.BRUSH_DEFAULT)
        self.setPen(Node.Pen.DEFAULT)

        self._is_highlighted = False
        self._tooltip = Node._TOOLTIP

    def setSize(self, size):
        self._radius = radius = size/2
        self.setRect(-radius, -radius, size, size)

    def setText(self, text):
        if text: self.label.setText(text)
        self.label.setVisible(bool(text))

    def setColor(self, color):
        self.setBrush(QBrush(QColor(color)) if color else Node.BRUSH_DEFAULT)

    def isHighlighted(self):
        return self._is_highlighted

    def setHighlighted(self, highlight):
        self._is_highlighted = highlight
        if not self.isSelected():
            self.itemChange(self.ItemSelectedChange, False)

    def itemChange(self, change, value):
        if change == self.ItemSelectedChange:
            self.setPen(Node.Pen.SELECTED if value else
                        Node.Pen.HIGHLIGHTED if self._is_highlighted else
                        Node.Pen.DEFAULT)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget):
        option.state &= ~QStyle.State_Selected  # We use a custom selection pen
        super().paint(painter, option, widget)

    def setTooltip(self, callback):
        assert not callback or callable(callback)
        self._tooltip = callback or Node._TOOLTIP

    def hoverEnterEvent(self, event):
        self.setToolTip(self._tooltip())

    def hoverLeaveEvent(self, event):
        self.setToolTip('');


class GraphView(QGraphicsView):

    positionsChanged = QtCore.pyqtSignal(np.ndarray, float)
    # Emitted when nodes' selected or highlighted state changes
    selectionChanged = QtCore.pyqtSignal()
    # Emitted when the relayout() animation finishes
    animationFinished = QtCore.pyqtSignal()


    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = []
        self.edges = []
        self._selection = []
        self._clicked_node = None
        self.is_animating = False
        self.legend = None
        self._pressed = False

        scene = QGraphicsScene(self)
        scene.setItemIndexMethod(scene.BspTreeIndex)
        scene.setBspTreeDepth(2)
        self.setScene(scene)
        self.setSceneRect(-1e5, -1e5, 2e5, 2e5)
        self.scaleFactor = 300
        self.setText('')

        self.setCacheMode(self.CacheBackground)
        self.setViewportUpdateMode(self.FullViewportUpdate)  # BoundingRectViewportUpdate doesn't work on Mac

        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setResizeAnchor(self.AnchorViewCenter)

        self.setDragMode(self.RubberBandDrag)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.positionsChanged.connect(self._update_positions, type=Qt.BlockingQueuedConnection)
        self.animationFinished.connect(self.finish)

    def finish(self):
        self.is_animating = False
        self.centerView()

    def centerView(self):
        xs, ys = zip(*((item.x(), item.y())
                       for item in self.scene().items()
                       if isinstance(item, Node)))
        mx, Mx = min(xs), max(xs)
        my, My = min(ys), max(ys)
        w, h = Mx-mx, My-my
        self.centerOn(mx+w/2, my+h/2)

    def mousePressEvent(self, event):
        self._selection = []
        if event.button() == Qt.LeftButton:
            if self.is_animating:
                self.is_animating = False
                return
            # Save the current selection and restore it on mouse{Move,Release}
            self._clicked_node = self.itemAt(event.pos())
            self._pressed = True
            if self._clicked_node and isinstance(self._clicked_node, Node):
                self.setCursor(Qt.ClosedHandCursor)
            if event.modifiers() & Qt.ShiftModifier:
                self._selection = self.scene().selectedItems()
        # On right mouse button, switch to pan mode
        elif event.button() == Qt.RightButton:
            self.setDragMode(self.ScrollHandDrag)
            # Forge left mouse button event
            event = QtGui.QMouseEvent(event.type(),
                                      event.pos(),
                                      event.globalPos(),
                                      Qt.LeftButton,
                                      event.buttons(),
                                      event.modifiers())
        super().mousePressEvent(event)
        # Reselect the selection that had just been discarded
        for node in self._selection: node.setSelected(True)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if not self._clicked_node:
            for node in self._selection: node.setSelected(True)
        if not self._pressed:
            self.setCursor(
                Qt.OpenHandCursor if isinstance(self.itemAt(event.pos()), Node) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._pressed = False
        if self.dragMode() == self.RubberBandDrag:
            for node in self._selection: node.setSelected(True)
            self.selectionChanged.emit()
        if self._clicked_node and isinstance(self._clicked_node, Node):
            self.selectionChanged.emit()
            self.setCursor(Qt.OpenHandCursor)
        # The following line is required (QTBUG-48443)
        self.setDragMode(self.NoDrag)
        # Restore default drag mode
        self.setDragMode(self.RubberBandDrag)

    def setText(self, text):
        text = self._text = QtGui.QStaticText(text or '')
        text.setPerformanceHint(text.AggressiveCaching)
        option = QtGui.QTextOption()
        option.setWrapMode(QtGui.QTextOption.NoWrap)
        text.setTextOption(option)
        scene = self.scene()
        scene.invalidate(layers=scene.BackgroundLayer)

    def drawForeground(self, painter, rect):
        painter.resetTransform()
        painter.drawStaticText(10, 10, self._text)
        super().drawForeground(painter, rect)

    def scrollContentsBy(self, dx, dy):
        scene = self.scene()
        scene.invalidate(layers=scene.BackgroundLayer)
        super().scrollContentsBy(dx, dy)

    def _setState(self, nodes, extend, state_setter):
        nodes = set(nodes)
        if extend:
            for node in self.nodes:
                if node.id in nodes:
                    getattr(node, state_setter)(True)
        else:
            for node in self.nodes:
                getattr(node, state_setter)(node.id in nodes)
        self.selectionChanged.emit()

    def getSelected(self):
        return [node.id for node in self.scene().selectedItems()]

    def getUnselected(self):
        return [node.id
                for node in (set(self.scene().items()) - set(self.scene().selectedItems()))
                if isinstance(node, Node)]

    def setSelected(self, nodes, extend=False):
        self._setState(nodes, extend, 'setSelected')

    def getHighlighted(self):
        return [node.id for node in self.nodes
                if node.isHighlighted() and not node.isSelected()]

    def setHighlighted(self, nodes):
        self._setState(nodes, False, 'setHighlighted')

    def clear(self):
        self.scene().clear()
        self.scene().setSceneRect(QRectF())
        self.legend = None
        self.nodes.clear()
        self.edges.clear()

    def set_graph(self, graph, relayout=True):
        assert not graph or isinstance(graph, nx.Graph)
        self.graph = graph
        if not graph:
            self.clear()
            return
        large_graph = IS_LARGE_GRAPH(graph)
        very_large_graph = IS_VERY_LARGE_GRAPH(graph)
        self.setViewport(QtOpenGL.QGLWidget()
                         # FIXME: Try reenable the following test after Qt5 port
                         if large_graph and HAVE_OPENGL else
                         QWidget())
        self.setRenderHints(QtGui.QPainter.RenderHint() if very_large_graph else
                            (QtGui.QPainter.Antialiasing |
                             QtGui.QPainter.TextAntialiasing))
        self.clear()
        nodes = {}
        for v in sorted(graph.nodes()):
            node = Node(v, view=self)
            self.addNode(node)
            nodes[v] = node
        for u, v in graph.edges():
            self.addEdge(Edge(nodes[u], nodes[v], view=self))
        self.selectionChanged.emit()
        if relayout: self.relayout()

    def addNode(self, node):
        assert isinstance(node, Node)
        self.nodes.append(node)
        self.scene().addItem(node)

    def addEdge(self, edge):
        assert isinstance(edge, Edge)
        self.edges.append(edge)
        self.scene().addItem(edge)

    def wheelEvent(self, event):
        if event.angleDelta().x() != 0: return
        self.scaleView(2**(event.angleDelta().y() / 240))

    def scaleView(self, factor):
        magnitude = self.transform().scale(factor, factor).mapRect(QRectF(0, 0, 1, 1)).width()
        if 0.2 < magnitude < 30:
            self.scale(factor, factor)
        # Reposition nodes' labela and edges, both of which are node-dependend
        # (and nodes just "moved")
        for node in self.nodes: node.adjust()
        for edge in self.edges: edge.adjust()

    @property
    def is_animating(self):
        return self._is_animating

    @is_animating.setter
    def is_animating(self, value):
        self.setCursor(Qt.ForbiddenCursor if value else Qt.ArrowCursor)
        self._is_animating = value

    def relayout(self, randomize=True, weight=None):
        if self.is_animating: return
        self.is_animating = True
        if weight is None: weight = 'weight'
        pos, graphview = None, self
        if not randomize:
            pos = np.array([[pos.x(), pos.y()]
                            for pos in (node.pos()/self.scaleFactor for node in self.nodes)])

        class AnimationThread(Thread):
            def __init__(self, iterations, callback):
                super().__init__()
                self.daemon = True
                self.iterations = iterations
                self.callback = callback
            def run(self):
                newpos = fruchterman_reingold_layout(graphview.graph,
                                                     pos=pos,
                                                     weight=weight,
                                                     iterations=self.iterations,
                                                     sample_ratio=0.1,
                                                     callback=self.callback,
                                                     callback_rate=0.25)
                graphview.update_positions(newpos)
                graphview.animationFinished.emit()

        iterations, callback = FR_ITERATIONS, self.update_positions
        if IS_VERY_LARGE_GRAPH(self.graph):
            # Don't animate very large graphs
            iterations, callback = 5, None
        AnimationThread(iterations, callback).start()

    def update_positions(self, positions, progress=1.0):
        self.positionsChanged.emit(positions, progress)
        return self._is_animating

    def _update_positions(self, positions, _):
        for node, pos in zip(self.nodes, positions*self.scaleFactor):
            node.setPos(*pos)
        qApp.processEvents()


if __name__ == '__main__':
    import sys
    from AnyQt.QtWidgets import QApplication
    app = QApplication(sys.argv)
    widget = GraphView()
    widget.show()
    G = nx.scale_free_graph(int(sys.argv[1]) if len(sys.argv) > 1 else 100, seed=0)

    widget.set_graph(G)
    print('nodes', len(widget.nodes), 'edges', len(widget.edges))

    for node in widget.nodes[:10]:
        node.setHighlighted(True)

    sys.exit(app.exec_())
