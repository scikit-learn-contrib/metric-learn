# -*- coding: utf-8 -*-

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'numpydoc',
]

autodoc_default_flags = ['members', 'inherited-members']

default_role='any'

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

# General information about the project.
project = u'metric-learn'
copyright = u'2015-2017, CJ Carey and Yuan Tang'
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

