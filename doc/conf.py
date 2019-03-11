# -*- coding: utf-8 -*-
import sys

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx'
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

# General information about the project.
project = u'metric-learn'
copyright = u'2015-2018, CJ Carey and Yuan Tang'
author = u'CJ Carey and Yuan Tang'
version = '0.4.0'
release = '0.4.0'
language = 'en'

exclude_patterns = ['_build']
pygments_style = 'sphinx'
todo_include_todos = True
numpydoc_show_class_members = False

# Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'metric-learndoc'

# Option to only need single backticks to refer to symbols
default_role = 'any'

# Option to hide doctests comments in the documentation (like # doctest:
# +NORMALIZE_WHITESPACE for instance)
trim_doctest_flags = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None)
}