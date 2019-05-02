# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# We convert all SVG files to PDF for LaTeX output.  For HTML output, we don't
# create PNGs but rather use the SVG files directly.
FIGURESOURCES    = $(shell find source -name "*.svg")
GENERATEDPDF    = $(patsubst %.svg,%.pdf,$(FIGURESOURCES))

%.pdf: %.svg
	inkscape --export-pdf=$@ $<

# Emtpy target required so that the default target is not triggered
%.svg:


clean:
	rm -f $(GENERATEDPDF)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

latex: Makefile $(GENERATEDPDF)
	cd .. ; python setup.py build ; cd -
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	cd .. ; python setup.py build ; cd -
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
