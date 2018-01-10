Making a Kwant release
======================

This document guides a contributor through creating a release of Kwant.


Preflight checks
################

The following checks should be made *before* tagging the release.


Check that all issues are resolved
----------------------------------

Check that all the issues and merge requests for the appropriate
`milestone <https://gitlab.kwant-project.org/kwant/kwant/milestones>`
have been resolved. Any unresolved issues should have their milestone
bumped.


Ensure that all tests pass
--------------------------

This should be as simple as verifying that the latest CI pipeline succeeded.
For major and minor releases we will be tagging the ``master`` branch.
For patch releases, the ``stable`` branch.


Inspect the documentation
-------------------------

If the CI pipeline succeeded, then the latest docs should be available at:

    https://test.kwant-project.org/doc/<branch name>

Check that there are no glaring deficiencies.


Update the ``whatsnew`` file
----------------------------

For each new mior release, check that there is an appropriate ``whatsnew`` file
in ``doc/source/pre/whatsnew``.  This should be named as::

    <major>.<minor>.rst

and referenced from ``doc/source/pre/whatsnew/index.rst``.  It should contain a
list of the user-facing changes that were made in the release. With any luck
this file will have been updated after any major features were released, if not
then you can see what commits were introduced since the last release using
``git log``. You can also see what issues were assigned to the release's
milestons and get an idea of what was introduced from there.

Starting with Kwant 1.4, we also mention user-visible changes in bugfix
releases in the whatsnew files.


Verify that ``AUTHORS.rst`` is up-to-date
-----------------------------------------

The following command shows the number of commits per author since the last
annotated tag::

    t=$(git describe --abbrev=0); echo Commits since $t; git shortlog -s $t..


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

A particularly good way to expose hidden problems is building Debian packages
using an isolated minimal build environment (cowbuilder).  This approach is
described here.

Make an *annotated*, *signed* tag for the release. The tag must have the name::

    git tag -s v<version> -m "version <version>"

Be sure to respect the format of the tag name (leading "v", e.g. "v1.2.3").
The tag message format is the one that has been used so far.

Do *not* yet push the tag anywhere, it might have to be undone!


Build a source taball and inspect it
------------------------------------

    ./setup.py sdist

This creates the file dist/kwant-<version>.tar.gz.  It is a good idea to unpack it
in /tmp and inspect that builds in isolation and that the tests run::

    cd /tmp
    tar xzf ~/src/kwant/dist/kwant-<version>.tar.gz
    cd kwant-<version>
    ./setup.py test


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
    gbp checkout master

Now it is time to import the new source code.  There are two options.  If, as
recommended above, the tarball of the new version has not been made public yet,
it must be imported as follows::

    gbp import-orig ~/src/kwant/dist/kwant-<version>.tar.gz

Alternatively, the following commands will import the newest version from PyPI::

    uscan --report      # This will report if a newer version exists on PyPI
    gbp import-orig --uscan

Now it is time to review the patch queue.  Rebase and checkout the ``patch-queue/master`` branch using

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
Kwant, you will have to go back all the way to recreating the source tarball.  If the version to be packaged has been released publicly already, this will require a new bugfix version.


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

Build packports for the current Debian stable
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
reasonable sure that the release is ready to be published.

git
---

Push the tag to the official Kwant repository::

    git push origin v<version>

PyPI
----

PyPI (this requires a file ~/.pypirc with a vaild username and password)::

    twine upload -s dist/kwant-<version>.tar.gz

It is very important that the tarball uploaded here is the same (bit-by-bit,
not only the contents) as the one used for the Debian packaging.  Otherwise it
will not be possible to build the Debian package based on the tarball from
PyPI.

Kwant website
-------------

The tarball and its signature (generated by the twine command above) should be
also made available on the website::

    scp dist/kwant-<version>.tar.gz* kwant-website-downloads:downloads/kwant

Debian packages
---------------

Go to the Debian packaging repository and push out the changes::

    git push --tags origin master upstream

Now the Debian packages that we built previously need to be added to the
repository of Debian packages on the Kwant website.  So far this the full
version of this repository is kept on Christoph Groth's machine, so this
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

Now the source and binary Debian packages can be added.  The last line has to
be executed for all the .deb files and may be automated with a shell loop. (Be
sure to use the appropriate <dist>: for the above configuratoin file either
"testing" or "stretch-backports".)::

    reprepro includedsc <dist> ../../src/kwant_<version>-1.dsc
    reprepro includedeb <dist> python3-kwant_<version>-1_amd64.deb

Once all the packages have been added, upload the repository::

    rsync -avz --delete dists pool wfw:webapps/downloads/debian

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


Gather the autobuilt packages from CI
-------------------------------------

(This section needs to be updated.  Using the sdist package as generated by
PyPI requires pushing the tag to gitlab and as such is incompatible with doing
the Debian packaging with an unpublished tag.)

CI automatically generates:

+ HTML documentation
+ Zipped HTML documentation
+ PDF documentation
+ ``sdist`` package (for upload to PyPI)

These can be found in the artifacts of the last CI job in the pipeline,
``gather release artifacts``.


Publish to the Kwant website
----------------------------

(This section needs to be updated.  The twine tool creates the signature file
during the upload.)

To do the following you will need access to the webroots of ``downloads.kwant-project.org``
and ``kwant-project.org``. Ask Christoph Groth if you need to be granted access.

Take the tar archive in the ``dist`` directory of the CI artifacts and generate
a detached GPG signature::

    gpg --armor --detach-sign kwant-<major>.<minor>.<patch>.tar.gz

Take the archive and the ``.asc`` signature file that was just generated
and upload them to the ``kwant`` directory of ``downloads.kwant-project.org``.

Take the zip archive and the PDF in the ``docs`` directory of the CI artifacts
and upload them  to the ``doc`` directory of ``downloads.kwant-project.org``.
Point the symbolic links ``latest.zip`` and ``latest.pdf`` to these new files.

Take the ``docs/html`` directory of the CI artifacts and upload them to::

    doc/<major>.<minor>.<patch>/

on ``kwant-project.org``. Point the symbolic link ``<major>`` to this directory.


Publish to PyPI
---------------

(This also needs to be updated.)

Install `twine <https://pypi.python.org/pypi/twine>` and use it to upload
the tar archive in the ``sdist`` directory of the Ci artifacts downloaded
in the previous step::

    twine upload --sign -u <PyPI username> -p <PyPI password> sdist/*

the ``--sign`` flag signs the uploaded package with your default GPG key.
Ask Christoph Groth for the Kwant PyPI credentials.


Publish to Launchpad
--------------------



Publish to Conda forge
----------------------

Conda forge automates build/deploy by using CI on Github repositoried containing
recipes for making packages from their source distributions.

Fork the `Kwant feedstock <https://github.com/conda-forge/kwant-feedstock>`
repository and  edit the file ``recipe/meta.yml``. Correctly set the ``version``
at the top of the file. Set the ``sha256`` string in the ``source`` section
near the top of the file to the SHA256 hash of the kwant source distribution
that was uploaded to ``downloads.kwant-project.org``. This can be found by::

    sha256sum kwant-<major>.<minor>.<patch>.tar.gz

Now commit these changes and open a pull request on the Kwant feedstock
repository that includes your change. Ask Bas Nijholt or Joseph Weston
to review and accept the pull request, so that Kwant will be rebuilt.


Announce the release
--------------------

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
