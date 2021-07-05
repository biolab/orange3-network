"""
=======
Network
=======

Network visualization and analysis tools for Orange.

"""

# Category description for the widget registry
from orangecontrib.network import Network
from orangewidget.utils.signals import summarize, PartialSummary

NAME = "Network"

DESCRIPTION = "Network visualization and analysis tools for Orange."

BACKGROUND = "#C0FF97"

ICON = "icons/Category-Network.svg"


@summarize.register(Network)
def summarize_(net: Network):
    n = net.number_of_nodes()
    if len(net.edges) == 1:
        nettype = ['Network', 'Directed network'][net.edges[0].directed]
        details = f"<nobr>{nettype} with {n} nodes " \
                  f"and {net.number_of_edges()} edges</nobr>"
    else:
        details = f"<nobr>Network with {n} nodes"
        if net.edges:
            details += " and {len(net.edges)} edge types:</nobr><ul>" + "".join(
                f"<li>{len(edges)} edges, "
                f"{['undirected', 'directed'][edges.directed]}</li>"
                for edges in net.edges)

    return PartialSummary(n, details)
