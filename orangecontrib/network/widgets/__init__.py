"""
=======
Network
=======

Network visualization and analysis tools for Orange.

"""

# Category description for the widget registry
import sysconfig
from orangecontrib.network import Network
from orangewidget.utils.signals import summarize, PartialSummary

NAME = "Network"

DESCRIPTION = "Network visualization and analysis tools for Orange."

BACKGROUND = "#C0FF97"

ICON = "icons/Category-Network.svg"

# Location of widget help files.
WIDGET_HELP_PATH = (
    # Used for development.
    # You still need to build help pages using
    # make htmlhelp
    # inside doc folder
    ("{DEVELOP_ROOT}/doc/_build/html/index.html", None),

    # Documentation included in wheel
    # Correct DATA_FILES entry is needed in setup.py and documentation has to be built
    # before the wheel is created.
    ("{}/help/orange3-network/index.html".format(sysconfig.get_path("data")), None),

    # Online documentation url, used when the local documentation is available.
    # Url should point to a page with a section Widgets. This section should
    # includes links to documentation pages of each widget. Matching is
    # performed by comparing link caption to widget name.
    ("https://orange3-network.readthedocs.io/en/latest/", "")
)


@summarize.register
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
