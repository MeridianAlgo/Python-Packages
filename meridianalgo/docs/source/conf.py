# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from datetime import datetime
from meridianalgo import __version__

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "MeridianAlgo"
copyright = f"{datetime.now().year}, Meridian Algorithmic Research Team"
author = "Meridian Algorithmic Research Team"

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_mock_imports = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "yfinance",
    "requests",
    "pytz",
    "python-dateutil",
    "tqdm",
    "joblib",
    "ta",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Custom settings ---------------------------------------------------------

# Add custom CSS
html_css_files = [
    "css/custom.css",
]

# Add custom JavaScript
html_js_files = [
    "js/custom.js",
]

# Enable numpydoc style docstrings
numpydoc_show_class_members = False

# Enable todo extension
extensions.append("sphinx.ext.todo")
todo_include_todos = True

# Enable copy button for code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: |\s*\.\.\.: "
copybutton_prompt_is_regexp = True

# Enable autosummary
autosummary_generate = True

# Enable figure numbering
numfig = True

# Set master doc
master_doc = "index"
