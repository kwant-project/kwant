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
Check that there is an appropriate ``whatsnew`` file in ``doc/source/pre/whatsnew``.
This should be named as::

    <major>.<minor>.<patch>.rst

and should contain a list of the user-facing changes that were made in the
release. With any luck this file will have been updated after any major
features were released, if not then you can see what commits were introduced
since the last release using ``git log``. You can also see what issues were
assigned to the release's milestons and get an idea of what was introduced
from there.


Tag the release
###############
Make an *annotated*, *signed* tag for the release. The tag must have the semantic
versioning format::

    v<major>.<minor>.<patch>

Once the tag has been created, push it to Kwant Gitlab.


Post-tagging
############
Pushing a new tag to Kwant Gitlab will kick off a few CI jobs, but there is
still some manual work to do.


Prepare the Debian package
--------------------------


Gather the autobuilt packages from CI
-------------------------------------
CI automatically generates:

+ HTML documentation
+ Zipped HTML documentation
+ PDF documentation
+ ``sdist`` package (for upload to PyPI)

These can be found in the artifacts of the last CI job in the pipeline,
``gather release artifacts``.


Publish to the Kwant website
----------------------------
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
