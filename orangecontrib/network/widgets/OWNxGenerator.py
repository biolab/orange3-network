import numpy as np
import networkx as nx

from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Output

import orangecontrib.network as network


def _balanced_tree(n):
    """
    This function generates a balanced tree with approximately n nodes.
    Each node of this tree has r children. Number of nodes cannot
    be equal to n because real number of nodes is a sum of series
    1 + r + r^2 + ... + r^h and we do not want to generate a trivial network
    with one node and its n-1 children. Therefore this algorithm
    finds a closest possible match. Number of children is also determined
    by a logarithmic cost function.

    Args:
        n (int): number of nodes

    Returns:
        network x graph
    """
    series = lambda r, h: sum([r**x for x in range(h + 1)])
    h, r = 1, 2
    off = n
    for r_ in range(2, 10):
        last = series(r_, h)
        for h_ in range(1, 15):
            new = series(r_, h_)
            if abs(new - n) * np.log2(r_) < off:
                off = abs(new - n)
                r, h = r_, h_
            if abs(n - new) > abs(n - last):
                break
            last = new
    return nx.balanced_tree(int(r), int(h))


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
    REGULAR = ('Regular', lambda n: nx.random_regular_graph(min(np.random.randint(10)*2, n - 1), n))
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


class OWNxGenerator(widget.OWWidget):
    name = "Network Generator"
    description = "Construct example graphs."
    icon = "icons/NetworkGenerator.svg"
    priority = 6420

    class Outputs:
        network = Output("Network", network.Graph, replaces=["Generated network"])

    graph_type = settings.Setting(0)
    n_nodes = settings.Setting(50)
    auto_commit = settings.Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        gui.comboBox(self.controlArea, self, 'graph_type',
                     label='Network type:',
                     items=GraphType.all,
                     orientation='horizontal',
                     callback=self.generate)
        gui.spin(self.controlArea, self, 'n_nodes',
                 10, 99999, 10,
                 label='Approx. number of nodes:',
                 orientation='horizontal',
                 callbackOnReturn=True,
                 callback=self.generate)
        gui.auto_commit(self.controlArea, self, 'auto_commit',
                        label='Generate network',
                        checkbox_label='Auto-generate')
        self.commit()

    def generate(self):
        return self.commit()

    def commit(self):
        _, func = GraphType.all[self.graph_type]
        graph = network.readwrite._wrap(func(self.n_nodes))
        self.Outputs.network.send(graph)


if __name__ == "__main__":
    from AnyQt.QtWidgets import QApplication
    a = QApplication([])
    ow = OWNxGenerator()
    ow.show()
    a.exec_()
    ow.saveSettings()
