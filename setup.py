#!/usr/bin/env python

import os
import sys

from setuptools import find_packages

if sys.version_info < (3, 4):
    sys.exit('Orange3-Network requires Python >= 3.4')

from numpy.distutils.core import setup

NAME = 'Orange3-Network'
DOCUMENTATION_NAME = 'Orange Network'

VERSION = '1.2.0'

DESCRIPTION = 'Networks add-on for Orange 3 data mining software package.'
LONG_DESCRIPTION  = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
AUTHOR = 'Laboratory of Bioinformatics, FRI UL'
AUTHOR_EMAIL = 'miha.stajdohar@gmail.com'
URL = 'https://github.com/biolab/orange3-network'
LICENSE = 'GPLv3'

KEYWORDS = (
    'network',
    'network analysis',
    'network layout',
    'network visualization',
    'data mining',
    'machine learning',
    'artificial intelligence',
    'orange',
    'orange3 add-on',
)

CLASSIFIERS = (
    'Development Status :: 4 - Beta',
    'Environment :: X11 Applications :: Qt',
    'Environment :: Console',
    'Environment :: Plugins',
    'Programming Language :: Python',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
)

PACKAGES = find_packages(
    exclude = ('*.tests', '*.tests.*', 'tests.*', 'tests'),
)

PACKAGE_DATA = {
    'orangecontrib.network': ['networks/*'],
    'orangecontrib.network.widgets': ['icons/*']
}

SETUP_REQUIRES = (
    'setuptools',
)

INSTALL_REQUIRES = (
    'networkx>=1.10',
    'pyqtgraph>=0.9.10',
),

EXTRAS_REQUIRE = {
    # Dependencies which are problematic to install automatically
    'GUI': (
        'PyQt4',
    ),
    'reST': (
        'numpydoc',
    ),
}

DEPENDENCY_LINKS = (
)

ENTRY_POINTS = {
    'orange.addons': (
        'network = orangecontrib.network',
    ),
    'orange.widgets': (
        'Network = orangecontrib.network.widgets',
    ),
    'orange.data.io.search_paths': (
        'network = orangecontrib.network:networks',
    ),
    'orange.widgets': (
        'Networks = orangecontrib.network.widgets',
    )
}

NAMESPACES = ["orangecontrib"]

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('orangecontrib.network')
    return config


if __name__ == '__main__':
    setup(
        configuration=configuration,
        name = NAME,
        version = VERSION,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        url = URL,
        license = LICENSE,
        keywords = KEYWORDS,
        classifiers = CLASSIFIERS,
        packages = PACKAGES,
        package_data = PACKAGE_DATA,
        setup_requires = SETUP_REQUIRES,
        install_requires = INSTALL_REQUIRES,
        extras_require = EXTRAS_REQUIRE,
        dependency_links = DEPENDENCY_LINKS,
        entry_points = ENTRY_POINTS,
        namespace_packages=NAMESPACES,
        include_package_data = True,
        zip_safe = False,
    )
