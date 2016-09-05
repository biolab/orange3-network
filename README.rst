Orange3 Network
===============

Orange3-Network is an add-on for Orange_ data mining software package. It
provides network visualization and network analysis tools.

.. _Orange: http://orange.biolab.si/

Documentation is found at:

http://orange3-network.rtfd.io

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

Installation on Windows
-----------------------

Windows users have to compile the code. Run it with either Microsoft Visual Studio

    python setup.py build_ext -i --compiler=msvc install

or `Mingw32 <https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/rubenvb/gcc-4.8-release/>`__:

    python setup.py build_ext -i --compiler=mingw32 install

Usage
-----

After the installation, the widgets from this add-on are registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python3 -m Orange.canvas

New widgets are in the toolbox bar under Networks section.
