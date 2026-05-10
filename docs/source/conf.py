"""Sphinx configuration for DeconX documentation."""
from datetime import datetime

project = "DeconX"
author = "Yimin Fan and contributors"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "DeconX"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/Seniorious123/DeconX",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#2b6cb0",
        "color-brand-content": "#2b6cb0",
    },
    "dark_css_variables": {
        "color-brand-primary": "#63b3ed",
        "color-brand-content": "#63b3ed",
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
