from functools import reduce

import numpy as np

from AnyQt.QtWidgets import QSpinBox

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Output, Msg

from orangecontrib.network import Network
# __all__ is defined, pylint: disable=wildcard-import, unused-wildcard-import
from orangecontrib.network.network.generate import *


class GraphType:
    """
    BALANCED_TREE = ('Balanced tree', _balanced_tree)
    REGULAR = ('Regular', lambda n: nx.random_regular_graph(min(np.random.randint(10)*2, n - 1), n))
    SCALEFREE = ('Scale-free', lambda n: nx.scale_free_graph(int(n)))
    SHELL = ('Shell', lambda n: nx.random_shell_graph([(int(n*.1), int(n*.1), .2),
                                                       (int(n*.3), int(n*.3), .8),
                                                       (int(n*.6), int(n*.6), .5)]))
    WHEEL = ('Wheel', lambda n: nx.wheel_graph(int(n)))
"""


def _ctrl_name(name, arg):
    return (name + "___" + arg).replace(" ", "_")


class OWNxGenerator(widget.OWWidget):
    name = "Network Generator"
    description = "Construct example graphs."
    icon = "icons/NetworkGenerator.svg"
    priority = 6420

    GRAPH_TYPES = (
        ("Path", path, {"Path of length": 10}, ""),
        ("Cycle", cycle, {"Cycle with": 10}, "nodes"),
        ("Complete", complete, {"Complete with": 5}, "nodes"),
        ("Complete bipartite", complete_bipartite,
         {"Complete bipartite with": 5, "and": 8}, "nodes"),
        ("Barbell", barbell, {"Barbell with": 5, "and": 8}, "nodes"),
        ("Ladder", ladder, {"Ladder with": 10}, "steps"),
        ("Circular ladder", circular_ladder,
         {"Circular ladder with": 8}, "steps"),
        ("Grid", grid, {"Grid of height": 4, "and width": 5}, ""),
        ("Hypercube", hypercube, {"Hypercube,": 4}, "dimensional"),
        ("Star", star, {"Star with": 10}, "edges"),
        ("Lollipop", lollipop,
         {"Lollipop with": 5, "nodes, stem of": 5}, "nodes"),
        ("Geometric", geometric,
         {"Geometric with": 20, "nodes, ": 50}, "edges", True)
    )

    mins_maxs = {
        "Geometric": ((5, 1000), (20, 10000))
    }

    class Outputs:
        network = Output("Network", Network)

    class Error(widget.OWWidget.Error):
        generation_error = Msg("{}")

    graph_type = settings.Setting(7)
    arguments = settings.Setting(
        reduce(lambda x, y: {**x, **y},
               ({_ctrl_name(name, arg): val for arg, val in defaults.items()}
                for name, _, defaults, *_1 in GRAPH_TYPES)))

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        rb = self.radios = gui.radioButtons(
            self.controlArea, self, "graph_type",
            box="Graph type",
            callback=self.on_type_changed
        )
        self.arg_spins = {}
        for name, _, arguments, post, *_ in self.GRAPH_TYPES:
            argbox = gui.hBox(rb)
            gui.appendRadioButton(rb, name, argbox)
            self.arg_spins[name] = box = []
            min_max = self.mins_maxs.get(name, ((1, 100),) * len(arguments))
            for arg, (minv, maxv) in zip(arguments, min_max):
                box.append(gui.widgetLabel(argbox, arg))
                spin = QSpinBox(value=self.arguments[_ctrl_name(name, arg)],
                                minimum=minv, maximum=maxv)
                argbox.layout().addWidget(spin)
                spin.valueChanged.connect(
                    lambda value, name=name, arg=arg:
                    self.update_arg(value, name, arg))
                box.append(spin)
            if post:
                box.append(gui.widgetLabel(gui.hBox(argbox), post))

        self.bt_generate = gui.button(
            self.controlArea, self, "Regenerate Network",
            callback=self.generate)
        self.on_type_changed()

    def on_type_changed(self):
        cur_def = self.GRAPH_TYPES[self.graph_type]
        cur_name = cur_def[0]
        for (name, spins), radio in zip(self.arg_spins.items(), self.radios.buttons):
            radio.setText(name * (cur_name != name))
            for spin in spins:
                spin.setHidden(name != cur_name)

        is_random = len(cur_def) >= 5 and cur_def[4]
        self.bt_generate.setEnabled(is_random)

        self.generate()

    def update_arg(self, value, name, arg):
        self.arguments[_ctrl_name(name, arg)] = value
        self.generate()

    def generate(self):
        name, func, args, *_ = self.GRAPH_TYPES[self.graph_type]
        args = tuple(self.arguments[_ctrl_name(name, arg)] for arg in args)
        self.Error.generation_error.clear()
        try:
            network = func(*args)
        except ValueError as exc:
            self.Error.generation_error(exc)
            network = None
        else:
            n = len(network.nodes)
            network.nodes = Table(Domain([], [], [StringVariable("id")]),
                                  np.zeros((n, 0)), np.zeros((n, 0)),
                                  np.arange(n).reshape((n, 1)))
        self.Outputs.network.send(network)


def main():
    def send(graph):
        owe.set_graph(graph)
        owe.handleNewSignals()

    from AnyQt.QtWidgets import QApplication
    from orangecontrib.network.widgets.OWNxExplorer import OWNxExplorer
    a = QApplication([])
    ow = OWNxGenerator()
    owe = OWNxExplorer()
    ow.Outputs.network.send = send
    ow.show()
    owe.show()
    a.exec_()
    ow.saveSettings()


if __name__ == "__main__":
    main()
