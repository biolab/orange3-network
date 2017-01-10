Orange3 Network
===============

[![Documentation Status](https://readthedocs.org/projects/orange3-network/badge/?version=latest)](http://orange3-network.readthedocs.io/en/latest/?badge=latest)

Orange3-Network is an add-on for [Orange] data mining software package. It
provides network visualization and network analysis tools.

[Orange]: http://orange.biolab.si/

Documentation is found at: http://orange3-network.rtfd.io

Installation
------------

Install from Orange add-on installer through Options - Add-ons.

To install the add-on with pip use

    pip install orange3-network

To install the add-on from source, run

    python setup.py install

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    python setup.py develop

You can also run

    pip install -e .

which is sometimes preferable as you can *pip uninstall* packages later.

### Anaconda

If using Anaconda Python distribution, simply run

    pip install orange3-network

### Compiling on Windows

If you are not using Anaconda distribution, but building the add-on directly from the source code, Windows users need to compile the code.
Download [Microsoft Visual Studio compiler] and run the command

    python setup.py build_ext -i --compiler=msvc install

[Microsoft Visual Studio compiler]: http://landinghub.visualstudio.com/visual-cpp-build-tools

Usage
-----

After the installation, the widgets from this add-on are registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python3 -m Orange.canvas

New widgets are in the toolbox bar under Networks section.
