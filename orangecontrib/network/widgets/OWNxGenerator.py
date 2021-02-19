import string
from collections import defaultdict

import numpy as np

from AnyQt.QtCore import Qt
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
        (path, "Path",
         "of length", (10, 2, 100)),
        (cycle, "Cycle",
         "with", (10, 3, 100), "nodes"),
        (complete, "Complete",
         "with", (5, 1, 100), "nodes"),
        (complete_bipartite, "Complete bipartite",
         "with", (5, 1, 100), "and", (8, 1, 100), "nodes"),
        (barbell, "Barbell",
         "with", (5, 1, 100), "and", (8, 1, 100), "nodes"),
        (ladder, "Ladder",
         "with", (10, 2, 100), "steps"),
        (circular_ladder, "Circular ladder",
         "with", (8, 2, 100), "steps"),
        (grid, "Grid",
         ", height", (4, 2, 100), "and width", (5, 2, 100)),
        (hypercube, "Hypercube",
         ", ", (4, 1, 14), "dimensional"),
        (star, "Star", "with", (10, 1, 1000), "edges"),
        (lollipop, "Lollipop",
         "with", (5, 3, 30), "nodes, stem of", (5, 3, 100), "nodes"),
        (geometric, "Geometric",
         "with", (20, 2, 100), "nodes, ", (50, 1, 10000), "edges")
    )
    RANDOM_TYPES = ("Geometric", )

    class Outputs:
        network = Output("Network", Network)

    class Error(widget.OWWidget.Error):
        generation_error = Msg("{}")

    graph_type = settings.Setting(7)
    arguments = settings.Setting(
        {name: [args[0] for args in defaults if isinstance(args, tuple)]
         for _, name, *defaults in GRAPH_TYPES})
    settings_version = 2

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        def space(a, b):
            if not isinstance(a, str) or a[-1] in string.ascii_letters \
                    and not isinstance(b, str) or b[0] in string.ascii_letters:
                return " "
            else:
                return ""

        super().__init__()
        rb = self.radios = gui.radioButtons(
            self.controlArea, self, "graph_type",
            box=True,
            callback=self.on_type_changed
        )
        rb.layout().setSpacing(6)
        self.arg_spins = {}
        for (_1, name, _2, *arguments), defaults \
                in zip(self.GRAPH_TYPES, self.arguments.values()):
            argbox = gui.hBox(rb)
            argbox.layout().setSpacing(0)
            rbb = gui.appendRadioButton(rb, name, argbox)
            rbb.setAttribute(Qt.WA_LayoutUsesWidgetRect)
            argbox.setAttribute(Qt.WA_LayoutUsesWidgetRect)
            self.arg_spins[name] = box = []
            values = iter(defaults)
            argno = 0
            for prev, arg, post in \
                    zip([""] + arguments, arguments, arguments + [""]):
                if isinstance(arg, str):
                    label = space(prev, arg) + arg + space(arg, post)
                    box.append(gui.widgetLabel(argbox, label))
                else:
                    assert isinstance(arg, tuple)
                    _value, minv, maxv = arg
                    spin = QSpinBox(value=next(values), minimum=minv, maximum=maxv)
                    argbox.layout().addWidget(spin)
                    spin.valueChanged.connect(
                        lambda value, name=name, argidx=argno:
                        self.update_arg(value, name, argidx))
                    argno += 1
                    box.append(spin)

        self.bt_generate = gui.button(
            self.controlArea, self, "Regenerate Network",
            callback=self.generate)
        self.on_type_changed()

    def on_type_changed(self):
        cur_def = self.GRAPH_TYPES[self.graph_type]
        cur_name = cur_def[1]
        for (name, widgets), (_1, _2, name_add, *_), radio in \
                zip(self.arg_spins.items(), self.GRAPH_TYPES, self.radios.buttons):
            selected = name == cur_name
            radio.setText(name
                + (" " * (name_add and name_add[0] in string.ascii_letters)
                   + name_add) * selected)
            for spin in widgets:
                spin.setHidden(not selected)

        self.bt_generate.setEnabled(cur_name in self.RANDOM_TYPES)

        self.generate()

    def update_arg(self, value, name, argidx):
        self.arguments[name][argidx] = value
        self.generate()

    def generate(self):
        func, name, args, *_ = self.GRAPH_TYPES[self.graph_type]
        args = self.arguments[name]
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

    @classmethod
    def migrate_settings(cls, settings_, version):
        if version < 2:
            arguments = defaultdict(list)
            for name, value in settings_.pop("arguments", {}):
                arguments[name.split("__")[0]].append(value)
            settings_["arguments"] = dict(arguments)

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
    owe.show()
    ow.show()
    a.exec_()
    ow.saveSettings()


if __name__ == "__main__":
    main()
