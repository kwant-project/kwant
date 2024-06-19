# Release %vX.Y.Z **UPDATE THIS**

This is a checklist based on the [release document](RELEASE.rst); consult it for more details.

## Preflight checks

- [ ] All the issues from this milestone are resolved
- [ ] Ensure that tests pass on [main](https://gitlab.kwant-project.org/kwant/kwant/tree/main) branch
- [ ] Documentation looks correct https://test.kwant-project.org/doc/main
- [ ] [Whatnew](doc/source/pre/whatsnew) is up to date
- [ ] `AUTHORS.rst` and `.mailmap` are up to date (run `git shortlog -s | sed -e "s/^ *[0-9\t ]*//"| xargs -i sh -c 'grep -q "{}" AUTHORS.rst || echo "{}"'`)

## Make a release, but do not publish it yet

- [ ] Tag the release
- [ ] Build the source tarball and inspect it
- [ ] Build the documentation

## Test packaging

These steps may be done in parallel

### Debian

- [ ] Follow the steps for building the Debian packages from [RELEASE.rst](RELEASE.rst)

### Conda

- [ ] Publish the release candidate source tarball "somewhere on the internet" (SOTI)
- [ ] open a PR to the [conda-forge/kwant-feedstock](https://github.com/conda-forge/kwant-feedstock/) repo on Github. Make sure to mark the PR as WIP so that it doesn't get merged in accidentally
    - [ ] set the upstream tarball URL (in meta.yaml) to SOTI
    - [ ] update the tarball hash in meta.yaml
- [ ] See if the package builds

## Publish the release
- [ ] push the tag
- [ ] upload the source tarball to PyPI
- [ ] upload the source tarball to the Kwant website
- [ ] publish the debian packages
- [ ] publish the ubuntu packages
- [ ] create a new version of the Kwant conda-forge feedstock, and open a pull request to upstream
- [ ] upload the documentation to the Kwant website
- [ ] update the Kwant website to say that Conda is the preferred way to install Kwant on Windows

## Announce the release
- [ ] Write a short post summarizing the highlights on the Kwant website
- [ ] post to the Kwant mailing list

## Working towards next release
- [ ] add a whatsnew file for the next release
- [ ] tag this release with an `a0` suffix
- [ ] push this tag to the official Kwant repository
- [ ] create a milestone for the next release
