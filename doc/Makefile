# Makefile for Sphinx documentation

# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python3 -c 'import sys, sphinx; sys.exit(sphinx.main(sys.argv))'
PAPER         =
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

# We convert all SVG files to PDF for LaTeX output.  For HTML output, we don't
# create PNGs but rather use the SVG files directly.
FIGURESOURCES    = $(shell find source -name "*.svg")
GENERATEDPDF    = $(patsubst %.svg,%.pdf,$(FIGURESOURCES))

# Figure generation from patched tutorial scripts
#
# As make does not support the generation of multiple targets by a single
# invocation of a (non-implicit) rule, we use a trick: We pretend to be
# generating a single (empty) flag file per invocation.  The figure files are
# generated as well, but only as side-effects.  Each flag file is used to
# remember the time at which the corresponding figure-generating script was run.
# This works perfectly unless the actual output files are deleted without
# deleting the corresponding flag file.
FIGSCRIPTS = $(patsubst %.diff,%,$(notdir $(wildcard source/code/figure/*.py.diff)))
FIGURES = $(patsubst %.py,source/code/figure/.%_flag,$(FIGSCRIPTS))
SCRIPTS = $(sort $(FIGSCRIPTS) $(notdir $(wildcard source/code/include/*.py)))
INCLUDES = $(patsubst %,source/code/include/%,$(SCRIPTS))
DOWNLOADS = $(patsubst %,source/code/download/%,$(SCRIPTS))

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
	-rm -f $(FIGURES)
	-rm -f $(patsubst %,source/code/include/%,$(FIGSCRIPTS))
	-rm -f $(DOWNLOADS)
	-rm -f $(patsubst %,source/code/figure/%,$(FIGSCRIPTS))
	-rm -f $(patsubst %.py,source/code/figure/%_*.png,$(FIGSCRIPTS))
	-rm -f $(patsubst %.py,source/code/figure/%_*.pdf,$(FIGSCRIPTS))

html:	$(FIGURES) $(INCLUDES) $(DOWNLOADS)
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: $(FIGURES) $(INCLUDES) $(DOWNLOADS)
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

pickle: $(FIGURES) $(INCLUDES) $(DOWNLOADS)
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:   $(FIGURES) $(INCLUDES) $(DOWNLOADS)
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: $(FIGURES) $(INCLUDES) $(DOWNLOADS)
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp: $(FIGURES) $(INCLUDES) $(DOWNLOADS)
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/kwant.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/kwant.qhc"

latex:  $(GENERATEDPDF) $(FIGURES) $(INCLUDES) $(DOWNLOADS)
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
	inkscape --export-pdf=$@ $<

# Make tutorial scripts by extracting the (complete!) context of the "patches".
# We make sure not to use 'wiggle' here.
.SECONDARY:
source/code/include/%.py: source/code/figure/%.py.diff
	@sed -n '/^[- ]/ s/^.//p' <$< >$@
	@touch -r $< $@

source/code/download/%.py: source/code/include/%.py
	@mkdir -p source/code/download
	@grep -v '^#HIDDEN' <$< >$@

# Make the figure generation scripts by patching tutorial scripts.  If the
# tutorial scripts haven't been modified, don't patch but directly extract the
# figure generation scripts.  This means that 'wiggle' is only needed when the
# tutorial scripts have been modified.
.SECONDARY:
source/code/figure/%.py: source/code/include/%.py
	@if [ $< -nt $@.diff ]; then \
	    cp $< $@; \
	    rm -f $@.porig; \
	    if ! wiggle --replace $@ $@.diff; then \
	        command -v wiggle >/dev/null 2>&1 && \
	        echo "Resolve conflicts by editing the files named below"; \
	        touch -d@0 $@; \
	        exit 1; \
	    fi \
	else \
	    sed -n '/^[+ ]/ s/^.//p' <$@.diff >$@; \
	    touch -r $@.diff $@; \
	fi

# Make the figure generation scripts also depend on the diffs.
define makedep
source/code/figure/$(1): source/code/figure/$(1).diff
endef
$(foreach name,$(FIGSCRIPTS),$(eval $(call makedep,$(name))))

# Run an figure generation script.  When successful, and if the script is newer
# than the corresponding diff, recreate the latter.  Note that the
# corresponding tutorial script cannot be newer, since if it is, the figure
# generation script is generated from it by patching.
source/code/figure/.%_flag: source/code/figure/%.py
	cd $(dir $<) && python3 $(notdir $<)
	@if [ ! -f $<.diff -o $< -nt $<.diff ]; then \
	    wiggle --diff --lines source/code/include/$(notdir $<) $< >$<.diff; \
	    touch -r $< $<.diff; \
	fi
	@rm -f $<.porig
	@touch $@
