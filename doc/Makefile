# Makefile for Sphinx documentation
#

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

# Tutorial images.
TUTORIAL1A_IMAGES = source/images/1-quantum_wire_result.png source/images/1-quantum_wire_result.pdf source/images/1-quantum_wire_sys.png source/images/1-quantum_wire_sys.pdf
TUTORIAL2A_IMAGES = source/images/2-spin_orbit_result.png source/images/2-spin_orbit_result.pdf
TUTORIAL2B_IMAGES = source/images/2-quantum_well_result.png source/images/2-quantum_well_result.pdf
TUTORIAL2C_IMAGES = source/images/2-ab_ring_result.png source/images/2-ab_ring_result.pdf source/images/2-ab_ring_sys.png source/images/2-ab_ring_sys.pdf source/images/2-ab_ring_note1.png source/images/2-ab_ring_note1.pdf source/images/2-ab_ring_note2.png source/images/2-ab_ring_note2.pdf
TUTORIAL3A_IMAGES = source/images/3-band_structure_result.png source/images/3-band_structure_result.pdf
TUTORIAL3B_IMAGES = source/images/3-closed_system_result.png source/images/3-closed_system_result.pdf source/images/3-closed_system_sys.png source/images/3-closed_system_sys.pdf
TUTORIAL4_IMAGES = source/images/4-graphene_result.png source/images/4-graphene_result.pdf source/images/4-graphene_sys1.png source/images/4-graphene_sys1.pdf source/images/4-graphene_sys2.png source/images/4-graphene_sys2.pdf source/images/4-graphene_bs.png source/images/4-graphene_bs.pdf
TUTORIAL5A_IMAGES = source/images/5-superconductor_band_structure_result.png source/images/5-superconductor_band_structure_result.pdf
TUTORIAL5B_IMAGES = source/images/5-superconductor_transport_result.png source/images/5-superconductor_transport_result.pdf
ALL_TUTORIAL_IMAGES = $(TUTORIAL1A_IMAGES)  $(TUTORIAL2A_IMAGES) $(TUTORIAL2B_IMAGES) $(TUTORIAL2C_IMAGES) $(TUTORIAL3A_IMAGES) $(TUTORIAL3B_IMAGES) $(TUTORIAL4_IMAGES) $(TUTORIAL5A_IMAGES) $(TUTORIAL5B_IMAGES)

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
	-rm -f $(ALL_TUTORIAL_IMAGES) source/images/.*_flag

html:	$(ALL_TUTORIAL_IMAGES)
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: $(ALL_TUTORIAL_IMAGES)
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

pickle: $(ALL_TUTORIAL_IMAGES)
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:   $(ALL_TUTORIAL_IMAGES)
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: $(ALL_TUTORIAL_IMAGES)
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp: $(ALL_TUTORIAL_IMAGES)
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/kwant.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/kwant.qhc"

latex:  $(GENERATEDPDF) $(ALL_TUTORIAL_IMAGES)
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

# Generation of tutorial images.  This requires some serious make trickery, see
# http://article.gmane.org/gmane.comp.gnu.make.general/5806
$(TUTORIAL1A_IMAGES): source/images/.1-quantum_wire_flag
	@:
source/images/.1-quantum_wire_flag: source/images/1-quantum_wire.py
	cd source/images/ && python 1-quantum_wire.py
	touch source/images/.1-quantum_wire_flag

$(TUTORIAL2A_IMAGES): source/images/.2-spin_orbit_flag
	@:
source/images/.2-spin_orbit_flag: source/images/2-spin_orbit.py
	cd source/images/ && python 2-spin_orbit.py
	touch source/images/.2-spin_orbit_flag

$(TUTORIAL2B_IMAGES): source/images/.2-quantum_well_flag
	@:
source/images/.2-quantum_well_flag: source/images/2-quantum_well.py
	cd source/images/ && python 2-quantum_well.py
	touch source/images/.2-quantum_well_flag

$(TUTORIAL2C_IMAGES): source/images/.2-ab_ring_flag
	@:
source/images/.2-ab_ring_flag: source/images/2-ab_ring.py
	cd source/images/ && python 2-ab_ring.py
	touch source/images/.2-ab_ring_flag

$(TUTORIAL3A_IMAGES): source/images/.3-band_structure_flag
	@:
source/images/.3-band_structure_flag: source/images/3-band_structure.py
	cd source/images/ && python 3-band_structure.py
	touch source/images/.3-band_structure_flag

$(TUTORIAL3B_IMAGES): source/images/.3-closed_system_flag
	@:
source/images/.3-closed_system_flag: source/images/3-closed_system.py
	cd source/images/ && python 3-closed_system.py
	touch source/images/.3-closed_system_flag

$(TUTORIAL4_IMAGES): source/images/.4-graphene_flag
	@:
source/images/.4-graphene_flag: source/images/4-graphene.py
	cd source/images/ && python 4-graphene.py
	touch source/images/.4-graphene_flag

$(TUTORIAL5A_IMAGES): source/images/.5-superconductor_band_structure_flag
	@:
source/images/.5-superconductor_band_structure_flag: source/images/5-superconductor_band_structure.py
	cd source/images/ && python 5-superconductor_band_structure.py
	touch source/images/.5-superconductor_band_structure_flag

$(TUTORIAL5B_IMAGES): source/images/.5-superconductor_transport_flag
	@:
source/images/.5-superconductor_transport_flag: source/images/5-superconductor_transport.py
	cd source/images/ && python 5-superconductor_transport.py
	touch source/images/.5-superconductor_transport_flag
