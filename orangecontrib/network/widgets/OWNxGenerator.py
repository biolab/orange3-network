
from PyQt4.QtGui import *
from PyQt4.QtCore import *

import numpy as np
import networkx as nx

import Orange
from Orange.widgets import gui, widget, settings

import orangecontrib.network as network


def _balanced_tree(n):
    b = max(np.log(n) / 2, 2)
    h = np.log((n * (b - 1)) + 1)/np.log(b) - 1
    return nx.balanced_tree(int(b), int(h))


def _hypercube(n):
    """Hypercube's nodes are cube coordinates, which isn't so nice"""
    G = nx.hypercube_graph(int(np.log2(n) + .1))
    G = nx.relabel_nodes(G, {n: int('0b' + ''.join(str(i) for i in n), 2)
                             for n in G.node})
    return G


class GraphType:
    BALANCED_TREE = ('Balanced tree', _balanced_tree)
    BARBELL = ('Barbell', lambda n: nx.barbell_graph(int(n*.4), int(n*.3)))
    CIRCULAR_LADDER = ('Circular ladder', lambda n: nx.circular_ladder_graph(int(n/2)))
    COMPLETE = ('Complete', lambda n: nx.complete_graph(int(n)))
    COMPLETE_BIPARTITE = ('Complete bipartite',
                          lambda n: nx.complete_bipartite_graph(int(n*.6), int(n*.4)))
    CYCLE = ('Cycle', lambda n: nx.cycle_graph(int(n)))
    GRID = ('Grid', lambda n: nx.grid_graph([int(np.sqrt(n))]*2))
    HYPERCUBE = ('Hypercube', _hypercube)
    LADDER = ('Ladder', lambda n: nx.ladder_graph(int(n/2)))
    LOBSTER = ('Lobster', lambda n: nx.random_lobster(int(n / (1 + .7 + .7*.5)), .7, .5))
    LOLLIPOP = ('Lollipop', lambda n: nx.lollipop_graph(int(n/2), int(n/2)))
    PATH = ('Path', lambda n: nx.path_graph(int(n)))
    REGULAR = ('Regular', lambda n: nx.random_regular_graph(np.random.randint(10)*2, n))
    SCALEFREE = ('Scale-free', lambda n: nx.scale_free_graph(int(n)))
    SHELL = ('Shell', lambda n: nx.random_shell_graph([(int(n*.1), int(n*.1), .2),
                                                       (int(n*.3), int(n*.3), .8),
                                                       (int(n*.6), int(n*.6), .5)]))
    STAR = ('Star', lambda n: nx.star_graph(int(n - 1)))
    WAXMAN = ('Waxman', lambda n: nx.waxman_graph(int(n)))
    WHEEL = ('Wheel', lambda n: nx.wheel_graph(int(n)))

    all = (BALANCED_TREE, BARBELL, CIRCULAR_LADDER, COMPLETE, COMPLETE_BIPARTITE,
           CYCLE, GRID, HYPERCUBE, LADDER, LOBSTER, LOLLIPOP, PATH, REGULAR,
           SCALEFREE, SHELL, STAR, WAXMAN, WHEEL)


class Output:
    NETWORK = 'Generated network'


class OWNxGenerator(widget.OWWidget):
    name = "Network Generator"
    description = "Generate example graphs."
    icon = "icons/NetworkGenerator.svg"
    priority = 6420

    outputs = [(Output.NETWORK, network.Graph),]

    graph_type = settings.Setting(0)
    n_nodes = settings.Setting(50)
    auto_commit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        gui.comboBox(self.controlArea, self, 'graph_type',
                     label='Generate graph:',
                     items=GraphType.all,
                     orientation='horizontal',
                     callback=self.generate)
        gui.spin(self.controlArea, self, 'n_nodes',
                 10, 500, 10,
                 label='Approx. number of nodes:',
                 orientation='horizontal',
                 callbackOnReturn=True,
                 callback=self.generate)
        gui.auto_commit(self.controlArea, self, 'auto_commit',
                        label='Generate graph',
                        checkbox_label='Auto-generate')
        self.commit()

    def generate(self):
        return self.commit()

    def commit(self):
        _, func = GraphType.all[self.graph_type]
        graph = network.readwrite._wrap(func(self.n_nodes))
        self.send(Output.NETWORK, graph)


if __name__ == "__main__":
    a = QApplication([])
    ow = OWNxGenerator()
    ow.show()
    a.exec_()
    ow.saveSettings()
