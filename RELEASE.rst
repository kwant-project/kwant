Making a Kwant release
======================

This document guides a contributor through creating a release of Kwant.


Preflight checks
################

The following checks should be made *before* tagging the release.


Check that all issues are resolved
----------------------------------

Check that all the issues and merge requests for the appropriate
`milestone <https://gitlab.kwant-project.org/kwant/kwant/milestones>`_
have been resolved. Any unresolved issues should have their milestone
bumped.


Ensure that all tests pass
--------------------------

For major and minor releases we will be tagging the ``master`` branch.
For patch releases, the ``stable`` branch.
This should be as simple as verifying that the latest CI pipeline succeeded,
however in ``stable`` branch also manually trigger CI task of building the
conda package and verify that it, too, succeeds.


Inspect the documentation
-------------------------

If the CI pipeline succeeded, then the latest docs should be available at:

    https://test.kwant-project.org/doc/<branch name>

Check that there are no glaring deficiencies.


Update the ``whatsnew`` file
----------------------------

For each new minor release, check that there is an appropriate ``whatsnew`` file
in ``doc/source/pre/whatsnew``.  This should be named as::

    <major>.<minor>.rst

and referenced from ``doc/source/pre/whatsnew/index.rst``.  It should contain a
list of the user-facing changes that were made in the release. With any luck
this file will have been updated at the same time as a feature was implemented,
if not then you can see what commits were introduced since the last release using
``git log``. You can also see what issues were assigned to the release's
milestones and get an idea of what was introduced from there.

Starting with Kwant 1.4, we also mention user-visible changes in bugfix
releases in the whatsnew files.


Verify that ``AUTHORS.rst`` and ``.mailmap`` are up-to-date
-----------------------------------------

The following command shows if there are any committers that are missing from
``AUTHORS.rst``::

  git shortlog -s | sed -e "s/^ *[0-9\t ]*//"| xargs -i sh -c 'grep -q "{}" AUTHORS.rst || echo "{}"'

If it outputs anything, then either add the new contributors to the list, or add
new identities of old contributors to the ``.mailmap``

Make a release, but do not publish it yet
#########################################

Various problems can surface only during the process of preparing a release and
make it necessary to fix the codebase.  It would be a pity to have to succeed
the freshly released version by a minor release just to correct a glitch that
was detected too late.  Therefore it is a good idea to pursue the release as
far as possible without announcing it, such that it can be undone and corrected
if necessary.  In the past tests that failed on the x86-32 architecture and
wrongly declared dependencies have been detected in this way.


Tag the release
---------------

Make an *annotated*, *signed* tag for the release. The tag must have the name::

    git tag -s v<version> -m "version <version>"

Be sure to respect the format of the tag name (leading "v", e.g. "v1.2.3").
The tag message format is the one that has been used so far.

Do *not* yet push the tag anywhere; it might have to be undone!


Build a source tarball and inspect it
-------------------------------------

    ./setup.py sdist

This creates the file dist/kwant-<version>.tar.gz.  It is a good idea to unpack it
in /tmp and inspect that builds in isolation and that the tests run::

    cd /tmp
    tar xzf ~/src/kwant/dist/kwant-<version>.tar.gz
    cd kwant-<version>
    ./setup.py test


Build the documentation
-----------------------
Building the documentation requires 'sphinx' and a Latex installation.
First build the HTML and PDF documentation::

    ./setup.py build
    cd doc
    make realclean
    make html latex SPHINXOPTS='-A website_deploy=True -n -W'
    cd doc/build/latex
    make all-pdf

Then create a zipped version of the HTML documentation and name the PDF
consistently, storing them, for example, in the "dist" directory along with the
source tarballs::

    ln -s `pwd`/doc/build/html /tmp/kwant-doc-<version>
    (cd /tmp/; zip -r kwant-doc-<version>.zip kwant-doc-<version>)
    mv /tmp/kwant-doc-<version>.zip dist

    mv doc/build/latex/kwant.pdf dist/kwant-doc-<version>.pdf


Clone the repository of the Kwant Debian package
------------------------------------------------

This step needs to be performed only once.  The cloned repository can be reused
for subsequent releases.

Clone the "kwant-debian" repository and go into its root directory.  If you
keep the Kwant source in "src/kwant", a good location for the Debian package
repository is "src/debian/kwant".  The packaging process creates many files
that are placed into the parent directory of the packaging repository, hence
having an additional directory level ("src/debian") is a good way to keep these
files separate::

    mkdir debian
    cd debian
    git clone ssh://git@gitlab.kwant-project.org:443/kwant/debian-kwant.git kwant
    cd kwant

Create a local upstream branch::

    git branch upstream origin/upstream

Add a remote for the repository that contains the previously created tag::

    git remote add upstream_repo ~/src/kwant

Make sure that::

    git config --get user.name
    git config --get user.email

show correct information.


Release a new version of the Kwant Debian package
-------------------------------------------------

Fetch packaging work (from origin) and the git tag created above (from
upstream_repo) into the packaging repo::

    git fetch --all

Make sure that the branches ``master`` and ``upstream`` are up-to-date::

    git checkout upstream
    git merge --ff-only origin/upstream
    git checkout master
    git merge --ff-only origin/master

Debian packages may include "quilt" patches that are applied on top of the
pristine tarball.  The tool `gbp pq` manages these patches as a git branch
``patch-queue/master.  Execute the following commands to (re)create
that branch based on the patches in ``debian/patches``::

    gbp pq --force import
    git checkout master

Now it is time to import the new source code.  There are two options.  If, as
recommended above, the tarball of the new version has not been made public yet,
it must be imported as follows::

    gbp import-orig ~/src/kwant/dist/kwant-<version>.tar.gz

Alternatively, the following commands will import the newest version from PyPI::

    uscan --report      # This will report if a newer version exists on PyPI
    gbp import-orig --uscan

Now it is time to review the patch queue.  Rebase and checkout the ``patch-queue/master`` branch using::

    gbp pq rebase

As ususal, the rebase might require manual intervention.  Once done, review all
the commits of the ``patch-queue/master`` branch.  Are all patches still
needed, should any be removed?  When done (even if no changes were needed), recreate the files in ``debian/patches`` using::

    gbp pq export

If ``git diff`` reports any changes, be sure to commit them.

Now is the right moment to verify and modify the packaging information inside
the ``debian/`` directory.  For example, are the dependencies and versions
stated in ``debian/control`` up-to-date?

When all changes are commited, it is time to finalize by updating the Debian
changelog file.  Add a point "New upstream release" if there was one, and
describe any other changes to the Debian *packaging*::

    DEBEMAIL=<your-email> gbp dch -R --commit --distribution testing

Now verify that the package builds with::

    git clean -i
    gbp buildpackage

This is *not* how the package should be built for distribution.  For that, see
the following two sections.

If problems surface that require changing the packaging, undo the changelog
commit, modify the packaging, and re-iterate.  If the problems require fixing
Kwant, you will have to go back all the way to recreating the source tarball.
If the version to be packaged has been released publicly already, this will require a new bugfix version.


Setup git-pbuilder to build Debian packages
-------------------------------------------

Pbuilder is a tool to build Debian packages in an isolated chroot.  This allows
to verify that the package indeed only has the declared dependencies.  It also
allows to cross-build packages for i386 on amd64.

The following describes how to setup git-pbuilder, see also
https://wiki.debian.org/git-pbuilder.  This procedure needs to be executed only
once for a Debian system.

Install the Debian package git-buildpackage.

As root, add the following lines to /etc/sudoers or /etc/sudoers.d/local

    Cmnd_Alias BUILD = /usr/sbin/cowbuilder

and

    user     ALL = SETENV: BUILD

Now create pbuilder images.  In the following, replace ``<dist>`` by the
current Debian testing codename, e.g. "buster"::

    ARCH=i386 DIST=<dist> git-pbuilder create
    ARCH=amd64 DIST=<dist> git-pbuilder create

If the packages to be built have special dependencies, use the trick described in https://wiki.debian.org/git-pbuilder#Using_Local_Packages


Build Kwant packages using git-pbuilder
---------------------------------------

Update the builder environment (again, replace ``<dist>`` with the name of the
current Debian testing)::

    ARCH=i386 DIST=<dist> git-pbuilder update
    ARCH=amd64 DIST=<dist> git-pbuilder update

Now build the packages.  First the i386 package.  The option "--git-tag" tags
and signs the tag if the build is successful.  In a second step, the package is
built for amd64, but only the architecture-dependent files (not the
documentation package)::

    gbp buildpackage --git-pbuilder --git-arch=i386 --git-dist=<dist> --git-tag
    gbp buildpackage --git-pbuilder --git-arch=amd64 --git-dist=<dist> --git-pbuilder-options='--binary-arch'

Another example: build source package only::

    gbp buildpackage --git-export-dir=/tmp -S


Build backports for the current Debian stable
---------------------------------------------

Create a changelog entry for the backport::

    DEBEMAIL=<your-email> dch --bpo

As shown above, run ``git-pbuilder update`` for the appropriate distribution
codename.

Build backported packages::

    gbp buildpackage --git-pbuilder --git-ignore-new --git-arch=i386 --git-dist=<dist>
    gbp buildpackage --git-pbuilder --git-ignore-new --git-arch=amd64 --git-dist=<dist> --git-pbuilder-options='--binary-arch'

Do not commit anything.


Publish the release
###################

If the Debian packages build correctly that means that all tests pass both on
i386 and amd64, and that no undeclared dependencies are needed.  We can be
reasonably sure that the release is ready to be published.


git
---

Push the tag to the official Kwant repository::

    git push origin v<version>


PyPI
----

Install `twine <https://pypi.python.org/pypi/twine>`_ and run the following
(this requires a file ~/.pypirc with a valid username and password: ask
Christoph Groth to add you as a maintainer on PyPI, if you are not already)::

    twine upload -s dist/kwant-<version>.tar.gz

It is very important that the tarball uploaded here is the same (bit-by-bit,
not only the contents) as the one used for the Debian packaging.  Otherwise it
will not be possible to build the Debian package based on the tarball from
PyPI.


Kwant website
-------------

The following requires ssh access to ``kwant-project.org`` (ask Christoph
Groth). The tarball and its signature (generated by the twine command above) should be
uploaded to the downloads section of the website::

    scp dist/kwant-<version>.tar.gz* kwant-project.org:webapps/downloads/kwant


Debian packages
---------------

Go to the Debian packaging repository and push out the changes::

    git push --tags origin master upstream

Now the Debian packages that we built previously need to be added to the
repository of Debian packages on the Kwant website.  So far the full
version of this repository is kept on Christoph Groth's machine, so these
instructions are for reference only.

Go to the reprepro repository directory and verify that the configuration file
"conf/distributions" looks up-to-date.  It should look something like this (be
sure to update the codenames and the versions)::

    Origin: Kwant project
    Suite: stretch-backports
    Codename: stretch-backports
    Version: 9.0
    Architectures: i386 amd64 source
    Components: main
    Description: Unofficial Debian package repository of http://kwant-project.org/
    SignWith: C3F147F5980F3535

    Origin: Kwant project
    Suite: testing
    Codename: buster
    Version: 10.0
    Architectures: i386 amd64 source
    Components: main
    Description: Unofficial Debian package repository of http://kwant-project.org/
    SignWith: C3F147F5980F3535

If the config had to be updated execute::

    reprepro --delete clearvanished
    reprepro export
    reprepro --delete createsymlinks

In addition to the above, if distributions were removed from the
configuration file the corresponding directories must be removed
manually from under the `dists` subdirectory.

Now the source and binary Debian packages can be added.  The last line has to
be executed for all the .deb files and may be automated with a shell loop. (Be
sure to use the appropriate <dist>: for the above configuratoin file either
"testing" or "stretch-backports".)::

    reprepro includedsc <dist> ../../src/kwant_<version>-1.dsc
    reprepro includedeb <dist> python3-kwant_<version>-1_amd64.deb

Once all the packages have been added, upload the repository::

    rsync -avz --delete dists pool kwant-project.org:webapps/downloads/debian


Ubuntu packages
---------------

Packages for Ubuntu are provided as a PPA (Personal Package Archive):
https://launchpad.net/~kwant-project/+archive/ubuntu/ppa

Make sure ~/.dput.cf has something like this::

    [ubuntu-ppa-kwant]
    fqdn = ppa.launchpad.net
    method = ftp
    incoming = ~kwant-project/ppa/ubuntu/
    login = anonymous
    allow_unsigned_uploads = 0

We will also use the following script (prepare_ppa_upload)::

    #!/bin/sh

    if [ $# -eq 0 ]; then
        echo -e "\nUsage: $(basename $0) lousy mourning2 nasty\n"
        exit
    fi

    version=`dpkg-parsechangelog --show-field Version`
    mv debian/changelog /tmp/changelog.$$

    for release in $@; do
        cp /tmp/changelog.$$ debian/changelog
        DEBEMAIL=christoph.groth@cea.fr dch -b -v "$version~$release" -u low 'Ubuntu PPA upload'
        sed -i -e "1,1 s/UNRELEASED/$release/" debian/changelog
        debuild -S -sa
    done

    mv /tmp/changelog.$$ debian/changelog

Make sure that the Debian package builds correctly and go to its directory.

Check https://wiki.ubuntu.com/Releases for the relevant releases (we want to
provide packages at least for the current LTS release and the newer non-LTS
releases) and execute::

    prepare_ppa_upload <dist0> <dist1> <dist2>

(if a second upload of the same Debian version is needed, something like "vivid2" instead of "vivid" can be used.)

Now the changes files are "put" to start the build process on the PPA servers::

    cd ..
    dput ubuntu-ppa-kwant *~*.changes


Clone the repository of the Kwant conda-forge package
-----------------------------------------------------

This step needs to be performed only once.  The cloned repository can be reused
for subsequent releases.

Clone the "Kwant feedstock" repository and go into its root directory.  If you
keep the Kwant source in "src/kwant", a good location for the Conda package
repository is "src/conda-forge/kwant"::

    cd ~/src
    mkdir conda-forge
    cd conda-forge
    git clone https://github.com/conda-forge/kwant-feedstock kwant
    cd kwant

Rename the default remote to ``upstream``::

    git remote rename origin upstream


Create a new version of the Kwant conda-forge package
-----------------------------------------------------

Edit the file ``recipe/meta.yml``. Correctly set the ``version``
at the top of the file to the version of this release. Set the ``sha256``
string in the ``source`` section near the top of the file to the SHA256 hash
of the kwant source tarball that we previously created. You can find the
SHA256 hash by running ``openssl sha256 <filename>`` on Linux and Mac OSX.

Commit your changes.


Conda forge
-----------
This step requires a GitHub account, as Conda forge packages are autobuilt
from repositories hosted on GitHub.

Fork the `Kwant feedstock <https://github.com/conda-forge/kwant-feedstock>`_
repository and add your fork as a remote to the copy that you previously cloned::

    cd ~/conda-forge/kwant
    git remote add myfork https://github.com/<your-gh-username>/kwant-feedstock

Push the changes that you previously commited to your fork::

    git push myfork master

Open a pull request to Kwant feedstock repository. Ask Bas Nijholt or
Joseph Weston to review and accept the pull request.


Documentation
-------------
The following requires ssh access to  ``kwant-project.org``.
Ask Christoph Groth if you need to be granted access.

Upload the zipped HTML and PDF documentation::

    scp dist/kwant-doc-<version>.zip kwant-project.org:webapps/downloads/doc
    scp dist/kwant-doc-<version>.pdf kwant-project.org:webapps/downloads/doc

Point the symbolic links ``latest.zip`` and ``latest.pdf`` to these new files::

    ssh kwant-project.org "cd webapps/downloads/doc; ln -s kwant-doc-<version>.zip latest.zip"
    ssh kwant-project.org "cd webapps/downloads/doc; ln -s kwant-doc-<version>.pdf latest.pdf"

Then upload the HTML documentation for the main website::

    rsync -rlv --delete doc/build/html/* kwant-project.org:webapps/kwant/doc/<short-version>

where in the above ``<short-version>`` is just the major and minor version numbers.

Finally point the symbolic link ``<major-version>`` to ``<short-version>``::

    ssh kwant-project.org "cd webapps/kwant/doc; ln -s <major> <short-version>"


Announce the release
####################

Write a short post summarizing the highlights of the release on the
`Kwant website <https://gitlab.kwant-project.org/kwant/website>`, then
post this to the mailing list kwant-discuss@kwant-project.org.


Working towards the next release
################################

After finalizing a release, a new ``whatsnew`` file should be created for
the *next* release, and this addition should be committed and tagged as::

    <new major>.<new minor>.<new patch>a0

This tag should be pushed to Kwant Gitlab, and a new milestone for the next
release should be created.
