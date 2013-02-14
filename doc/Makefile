# Makefile for Sphinx documentation

# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

# In difference to the original Makefile, we convert all SVG files to PDF for
# LaTeX output.  For HTML output, we don't create PNGs but rather use the SVG
# files directly.
IMAGESOURCES    = $(shell find source -name "*.svg")
GENERATEDPDF    = $(patsubst %.svg, %.pdf, $(IMAGESOURCES))

expand = $(foreach name, $(foreach fig, $(2), $(1)_$(fig)), \
source/images/$(name).png source/images/$(name).pdf)

# Generated images.
1_IMAGES = $(call expand, quantum_wire, result sys)
2A_IMAGES = $(call expand, spin_orbit, result)
2B_IMAGES = $(call expand, quantum_well, result)
2C_IMAGES = $(call expand, ab_ring, result sys note1 note2)
3A_IMAGES = $(call expand, band_structure, result)
3B_IMAGES = $(call expand, closed_system, result sys)
4_IMAGES = $(call expand, graphene, result sys1 sys2 bs)
5A_IMAGES = $(call expand, superconductor_band_structure, result)
5B_IMAGES = $(call expand, superconductor_transport, result)
ALL_IMAGES = $(1_IMAGES)  $(2A_IMAGES) $(2B_IMAGES) $(2C_IMAGES) $(3A_IMAGES) \
$(3B_IMAGES) $(4_IMAGES) $(5A_IMAGES) $(5B_IMAGES)

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

clean:
	-rm -rf $(BUILDDIR)/* $(GENERATEDPDF)
	-rm -rf source/reference/generated

realclean: clean
	-rm -f $(ALL_IMAGES) source/images/.*_flag source/images/[a-zA-Z]*.py

html:	$(ALL_IMAGES)
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: $(ALL_IMAGES)
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

pickle: $(ALL_IMAGES)
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:   $(ALL_IMAGES)
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: $(ALL_IMAGES)
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp: $(ALL_IMAGES)
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/kwant.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/kwant.qhc"

latex:  $(GENERATEDPDF) $(ALL_IMAGES)
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


# Make the image generation scripts by patching tutorial scipts.
.SECONDARY:
%.py: %.py.diff
	@grep -v '^#HIDDEN' source/tutorial/$(notdir $@) >$@
	@patch $@ $<

# Generation of tutorial images.  This requires some make trickery, see
# http://article.gmane.org/gmane.comp.gnu.make.general/5806

.%_flag: %.py
	cd $(dir $<) && python $(notdir $<)
	@touch $@

$(1_IMAGES): source/images/.quantum_wire_flag
	@:

$(2A_IMAGES): source/images/.spin_orbit_flag
	@:

$(2B_IMAGES): source/images/.quantum_well_flag
	@:

$(2C_IMAGES): source/images/.ab_ring_flag
	@:

$(3A_IMAGES): source/images/.band_structure_flag
	@:

$(3B_IMAGES): source/images/.closed_system_flag
	@:

$(4_IMAGES): source/images/.graphene_flag
	@:

$(5A_IMAGES): source/images/.superconductor_band_structure_flag
	@:

$(5B_IMAGES): source/images/.superconductor_transport_flag
	@:
