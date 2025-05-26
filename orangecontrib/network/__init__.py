from importlib.resources import files, as_file

def networks():
    path_in_package = files(__package__).joinpath('networks')
    with as_file(path_in_package) as dir_path:
        yield ('', dir_path)

from .network import *

from orangecontrib.network.network import community
