# Makefile for Sphinx documentation

# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python2 -c 'import sys, sphinx; sys.exit(sphinx.main(sys.argv))'
PAPER         =
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

# We convert all SVG files to PDF for LaTeX output.  For HTML output, we don't
# create PNGs but rather use the SVG files directly.
IMAGESOURCES    = $(shell find source -name "*.svg")
GENERATEDPDF    = $(patsubst %.svg,%.pdf,$(IMAGESOURCES))

# Image generation from patched tutorial scripts
#
# As make does not support the generation of multiple targets by a single
# invocation of a (non-implicit) rule, we use a trick: We pretend to be
# generating a single (empty) flag file per invocation.  The image files are
# generated as well, but only as side-effects.  Each flag file is used to
# remember the time at which the corresponding image-generating script was run.
# This works perfectly unless the actual output files are deleted without
# deleting the corresponding flag file.
SCRIPTS = $(patsubst source/images/%.diff,%,$(wildcard source/images/*.py.diff))
FLAGS = $(patsubst %.py,source/images/.%_flag,$(SCRIPTS))

.PHONY: help clean realclean html dirhtml pickle json htmlhelp qthelp latex changes linkcheck doctest

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  dirhtml   to make HTML files named index.html in directories"
	@echo "  pickle    to make pickle files"
	@echo "  json      to make JSON files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  qthelp    to make HTML files and a qthelp project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  doctest   to run all doctests embedded in the documentation (if enabled)"
	@echo
	@echo "Append SPHINXOPTS='-A website_deploy=True' to include web analytics code."

clean:
	-rm -rf $(BUILDDIR)/* $(GENERATEDPDF)
	-rm -rf source/reference/generated

realclean: clean
	-rm -f $(FLAGS)
	-rm -f $(patsubst %,source/images/%,$(SCRIPTS))
	-rm -f $(patsubst %.py,source/images/%_*.png,$(SCRIPTS))
	-rm -f $(patsubst %.py,source/images/%_*.pdf,$(SCRIPTS))

html:	$(FLAGS)
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: $(FLAGS)
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

pickle: $(FLAGS)
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:   $(FLAGS)
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: $(FLAGS)
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp: $(FLAGS)
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/kwant.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/kwant.qhc"

latex:  $(GENERATEDPDF) $(FLAGS)
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

%.pdf: %.svg
	rsvg-convert -f pdf -o $@ $<

# Make the image generation scripts by patching tutorial scipts.
.SECONDARY:
%.py: %.py.diff
	@grep -v '^#HIDDEN' source/tutorial/$(notdir $@) >$@
	@patch $@ $<

# The image generation scripts depend on their unpatched originals
define makedep
source/images/$(1): source/tutorial/$(1)
endef
$(foreach name,$(SCRIPTS),$(eval $(call makedep,$(name))))

# Generation of images
.%_flag: %.py
	cd $(dir $<) && python $(notdir $<)
	@touch $@
