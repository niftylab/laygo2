# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../..'))
#sys.path.insert(0, os.path.abspath('../../user_guide_kor')) 이거 있어도 저 경로에 있는 md파일끼리 서로 링크 안됨.....

# -- Project information -----------------------------------------------------

project = 'laygo2'
copyright = '2021, Niftylab at Hanyang University, South Korea'
author = 'niftylab'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    #'sphinx_rtd_theme',
    #'sphinxcontrib.napoleon',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'myst_parser',
    'autoclasstoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

intersphinx_mapping = {
  'pockets': ('https://pockets.readthedocs.io/en/latest/', None),
  'python': ('https://docs.python.org/3', None),
  'sphinx': ('http://sphinx.readthedocs.io/en/latest/', None),
  'numpy': ('https://numpy.org/doc/stable', None),
}

source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Output file base name for HTML help builder.
htmlhelp_basename = 'napoleondoc'

autoclass_content = 'both'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme ="pydata_sphinx_theme"
#html_theme ="sphinx_rtd_theme" # 'alabaster'
#html_theme ="default"
#html_logo = "https://niftylab.github.io/assets/img/nifty_logo.png"
html_title = "laygo2"

html_theme_options = {
  "github_url": "https://github.com/niftylab/laygo2",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/logo.png"
#html_css_files = ['css/custom.css']


# -- Autodoc configuration -----------------------------------------------------
autoclass_content = 'class'
autodoc_member_order = 'bysource'
#autodoc_default_flags = ['members']
autodoc_default_options = {
    #'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    #'undoc-members': True,
}

# -- Autoclasstoc configuration --------------------------------------------
autoclasstoc_sections = [
    'public-attrs',
    'public-methods',
]


