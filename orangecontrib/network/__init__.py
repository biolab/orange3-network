from pkg_resources import resource_filename

def networks():
    yield ('', resource_filename(__name__, 'networks'))


from .network import *

from orangecontrib.network.network import community
