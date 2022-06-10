#!/usr/bin/env python

import os
from setuptools import setup

from distutils.command.build_ext import build_ext
from distutils.core import Extension

from setuptools import find_packages

import numpy

try:
    from Cython.Distutils.build_ext import new_build_ext as build_ext
    have_cython = True
except ImportError:
    have_cython = False

NAME = 'Orange3-Network'
DOCUMENTATION_NAME = 'Orange Network'

VERSION = '1.7.0'

DESCRIPTION = 'Networks add-on for Orange 3 data mining software package.'
LONG_DESCRIPTION  = open(os.path.join(os.path.dirname(__file__),
                                      'README.pypi')).read()
AUTHOR = 'Laboratory of Bioinformatics, FRI UL'
AUTHOR_EMAIL = 'info@biolab.si'
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

PACKAGES = find_packages()

PACKAGE_DATA = {
    'orangecontrib.network': ['networks/*'],
    'orangecontrib.network.widgets': ['icons/*'],
    'orangecontrib.network.widgets.tests': ['networks/*'],
    'orangecontrib.network.tests': ['*']
}

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

SETUP_REQUIRES = (
    'setuptools',
)

INSTALL_REQUIRES = (
    'anyqt',
    'gensim',
    'Orange3>=3.31',
    'orange-widget-base',
    'scipy',
    'scikit-learn',
),

EXTRAS_REQUIRE = {
    # Dependencies which are problematic to install automatically
    'GUI': (
        'AnyQt',
    ),
    'reST': (
        'numpydoc',
    ),
    'test': (
        'coverage',
    ),
    'doc': (
        'sphinx', 'recommonmark', 'sphinx_rtd_theme'
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
    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.network.widgets:WIDGET_HELP_PATH',
    )
}

NAMESPACES = ['orangecontrib']

class build_ext_error(build_ext):
    def initialize_options(self):
        raise SystemExit(
            "Cannot compile extensions. numpy and cython are required to "
            "build Orange."
        )

def ext_modules():
    includes = [numpy.get_include()]
    libraries = []

    if os.name == 'posix':
        libraries.append("m")

    return [
        # Cython extensions. Will be automatically cythonized.
        Extension(
            "*",
            ["orangecontrib/network/network/layout/*.pyx"],
            include_dirs=includes,
            libraries=libraries,
        )
    ]


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('orangecontrib.network')
    return config


def include_documentation(local_dir, install_dir):
    global DATA_FILES

    doc_files = []
    for dirpath, _, files in os.walk(local_dir):
        doc_files.append(
            (
                dirpath.replace(local_dir, install_dir),
                [os.path.join(dirpath, f) for f in files]
            )
        )
    DATA_FILES.extend(doc_files)


if __name__ == '__main__':
    cmdclass = {}
    if have_cython:
        cmdclass["build_ext"] = build_ext
    else:
        # substitute a build_ext command with one that raises an error when
        # building. In order to fully support `pip install` we need to
        # survive a `./setup egg_info` without numpy so pip can properly
        # query our install dependencies
        cmdclass["build_ext"] = build_ext_error

    include_documentation('doc/_build/html', 'help/orange3-network')

    setup(
        configuration=configuration,
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=PACKAGES,
        ext_modules=ext_modules(),
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        setup_requires=SETUP_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        dependency_links=DEPENDENCY_LINKS,
        entry_points=ENTRY_POINTS,
        namespace_packages=NAMESPACES,
        include_package_data=True,
        zip_safe=False,
        cmdclass=cmdclass
    )
