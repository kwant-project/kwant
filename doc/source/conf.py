# -*- coding: utf-8 -*-
#
# Kwant documentation build configuration file, created by
# sphinx-quickstart on Tue Jan 11 09:39:28 2011.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys, os

from distutils.util import get_platform
sys.path.insert(0, "../../build/lib.{0}-{1}.{2}".format(
        get_platform(), *sys.version_info[:2]))
import kwant

# -- General configuration -----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

sys.path.insert(0, os.path.abspath('../sphinxext'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary',
              'sphinx.ext.todo', 'sphinx.ext.pngmath', 'numpydoc',
              'kwantdoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Kwant'
copyright = '2011-2015, C. W. Groth (CEA), M. Wimmer, A. R. Akhmerov, X. Waintal (CEA), and others'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = kwant.__version__

# The short X.Y version.
version = release[:len(release) - len(release.lstrip('012345679.'))].rstrip('.')

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# Do not show all class members automatically in the class documentation
numpydoc_show_class_members = False

# -- Options for HTML output ---------------------------------------------------

# http://stackoverflow.com/questions/9728292/creating-latex-math-macros-within-sphinx
pngmath_latex_preamble = r"""\newcommand{\bra}[1]{\left\langle#1\right|}
\newcommand{\ket}[1]{\left|#1\right>}
\newcommand{\braket}[2]{\left\langle#1|#2\right\rangle}
\newcommand{\ri}{\text{i}}
\newcommand{\rd}{\text{d}}
"""

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = 'kwantdoctheme'
html_theme_path = ['..']
html_theme_options = {'collapsiblesidebar': True}
html_style = 'kwant.css'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
html_use_modindex = False

# This is needed too.
html_domain_indices = False

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'kwantdoc'


# -- Options for LaTeX output --------------------------------------------------

# http://thread.gmane.org/gmane.comp.python.sphinx.devel/4220/focus=4238
latex_elements = {'papersize': 'a4paper',
                  'release': '',
                  'releasename': '',
                  'preamble':
r"""\makeatletter
  \fancypagestyle{normal}{
    \fancyhf{}
    \fancyfoot[LE,RO]{{\py@HeaderFamily\thepage}}
    \fancyfoot[LO]{{\py@HeaderFamily\nouppercase{\rightmark}}}
    \fancyfoot[RE]{{\py@HeaderFamily\nouppercase{\leftmark}}}
    \fancyhead[LE,RO]{{\py@HeaderFamily \@title}}
    \renewcommand{\headrulewidth}{0.4pt}
    \renewcommand{\footrulewidth}{0.4pt}
  }
\makeatother
""" + pngmath_latex_preamble}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
# We use "et al." as it is shorter and there's not much space left horizontally.
latex_documents = [
  ('index', 'kwant.tex', 'Kwant {0} documentation'.format(release),
   'C. W. Groth, M. Wimmer, A. R. Akhmerov, X. Waintal, et al.',
   'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# This is needed too.
latex_domain_indices = False

# -- Options for autodoc -------------------------------------------------------
# Generate stub pages for autosummary directives.
autosummary_generate = True

autoclass_content = "both"
autodoc_default_flags = ['show-inheritance']

# -- Teach Sphinx to document bound methods like functions ---------------------
import types
from sphinx.ext import autodoc

class BoundMethodDocumenter(autodoc.FunctionDocumenter):
    objtype = "boundmethod"
    directivetype = 'function'

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        # Return True iff `member` is a bound method.  Taken from
        # <http://stackoverflow.com/a/1260881>.
        return (isinstance(member, types.MethodType) and
                member.__self__ is not None and
                not issubclass(member.__self__.__class__, type) and
                member.__self__.__class__ is not type)

    def format_args(self):
        args = super(BoundMethodDocumenter, self).format_args()
        left, sep, right = args.partition('self, ')
        if left.endswith('('):
            args = left + right
        return args

def setup(app):
    app.add_autodocumenter(BoundMethodDocumenter)
