# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Dual Autodiff'
copyright = '2024, Xin Li'
author = 'Xin Li'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', # Auto-generate documentation from docstrings
    'sphinx.ext.viewcode', # Add links to source code
    'sphinx.ext.napoleon', # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.mathjax', # Render math equations using MathJax
    'sphinx.ext.intersphinx', # Cross-reference other projects' documentation
    'sphinx.ext.todo', # Add a todo list
    'sphinx.ext.coverage', # Check for missing documentation
    'sphinx.ext.githubpages', # Publish to GitHub Pages
    'nbsphinx', # Build Jupyter notebooks
    'IPython.sphinxext.ipython_console_highlighting' # Highlight IPython console code
]


templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True