# -*- coding: utf-8 -*-
import sys
import os
import warnings

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
copyright = (u'2015-2020, CJ Carey, Yuan Tang, William de Vazelhes, Aurélien '
             u'Bellet and Nathalie Vauquier')
author = (u'CJ Carey, Yuan Tang, William de Vazelhes, Aurélien Bellet and '
          u'Nathalie Vauquier')
version = '0.6.2'
release = '0.6.2'
language = 'en'

exclude_patterns = ['_build']
pygments_style = 'sphinx'
todo_include_todos = True

# Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'metric-learndoc'

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


# sphinx-gallery configuration
sphinx_gallery_conf = {
    # to generate mini-galleries at the end of each docstring in the API
    # section: (see https://sphinx-gallery.github.io/configuration.html
    # #references-to-examples)
    'doc_module': 'metric_learn',
    'backreferences_dir': os.path.join('generated'),
}

# generate autosummary even if no references
autosummary_generate = True


# Temporary work-around for spacing problem between parameter and parameter
# type in the doc, see https://github.com/numpy/numpydoc/issues/215. The bug
# has been fixed in sphinx (https://github.com/sphinx-doc/sphinx/pull/5976) but
# through a change in sphinx basic.css except rtd_theme does not use basic.css.
# In an ideal world, this would get fixed in this PR:
# https://github.com/readthedocs/sphinx_rtd_theme/pull/747/files
def setup(app):
  app.add_js_file('js/copybutton.js')
  app.add_css_file('css/styles.css')


# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')
