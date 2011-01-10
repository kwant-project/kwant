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
TUTORIAL1A_IMAGES = source/images/tutorial1a_result.png source/images/tutorial1a_result.pdf source/images/tutorial1a_sys.png source/images/tutorial1a_sys.pdf
TUTORIAL2A_IMAGES = source/images/tutorial2a_result.png source/images/tutorial2a_result.pdf
TUTORIAL2B_IMAGES = source/images/tutorial2b_result.png source/images/tutorial2b_result.pdf
TUTORIAL2C_IMAGES = source/images/tutorial2c_result.png source/images/tutorial2c_result.pdf source/images/tutorial2c_sys.png source/images/tutorial2c_sys.pdf source/images/tutorial2c_note1.png source/images/tutorial2c_note1.pdf source/images/tutorial2c_note2.png source/images/tutorial2c_note2.pdf
TUTORIAL3A_IMAGES = source/images/tutorial3a_result.png source/images/tutorial3a_result.pdf
TUTORIAL3B_IMAGES = source/images/tutorial3b_result.png source/images/tutorial3b_result.pdf source/images/tutorial3b_sys.png source/images/tutorial3b_sys.pdf
TUTORIAL4_IMAGES = source/images/tutorial4_result.png source/images/tutorial4_result.pdf source/images/tutorial4_sys1.png source/images/tutorial4_sys1.pdf source/images/tutorial4_sys2.png source/images/tutorial4_sys2.pdf source/images/tutorial4_bs.png source/images/tutorial4_bs.pdf
ALL_TUTORIAL_IMAGES = $(TUTORIAL1A_IMAGES)  $(TUTORIAL2A_IMAGES) $(TUTORIAL2B_IMAGES) $(TUTORIAL2C_IMAGES) $(TUTORIAL3A_IMAGES) $(TUTORIAL3B_IMAGES) $(TUTORIAL4_IMAGES)

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
$(TUTORIAL1A_IMAGES): source/images/.tutorial1a_flag
	@:
source/images/.tutorial1a_flag: source/images/tutorial1a.py
	cd source/images/ && python tutorial1a.py
	touch source/images/.tutorial1a_flag

$(TUTORIAL2A_IMAGES): source/images/.tutorial2a_flag
	@:
source/images/.tutorial2a_flag: source/images/tutorial2a.py
	cd source/images/ && python tutorial2a.py
	touch source/images/.tutorial2a_flag

$(TUTORIAL2B_IMAGES): source/images/.tutorial2b_flag
	@:
source/images/.tutorial2b_flag: source/images/tutorial2b.py
	cd source/images/ && python tutorial2b.py
	touch source/images/.tutorial2b_flag

$(TUTORIAL2C_IMAGES): source/images/.tutorial2c_flag
	@:
source/images/.tutorial2c_flag: source/images/tutorial2c.py
	cd source/images/ && python tutorial2c.py
	touch source/images/.tutorial2c_flag

$(TUTORIAL3A_IMAGES): source/images/.tutorial3a_flag
	@:
source/images/.tutorial3a_flag: source/images/tutorial3a.py
	cd source/images/ && python tutorial3a.py
	touch source/images/.tutorial3a_flag

$(TUTORIAL3B_IMAGES): source/images/.tutorial3b_flag
	@:
source/images/.tutorial3b_flag: source/images/tutorial3b.py
	cd source/images/ && python tutorial3b.py
	touch source/images/.tutorial3b_flag

$(TUTORIAL4_IMAGES): source/images/.tutorial4_flag
	@:
source/images/.tutorial4_flag: source/images/tutorial4.py
	cd source/images/ && python tutorial4.py
	touch source/images/.tutorial4_flag
