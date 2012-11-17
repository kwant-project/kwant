# !/bin/sh

# This script regenerates the .diff files in this directory.  It's these files
# that are kept under vesion control instead of the scripts themselves.

for f in [0-9]-*.py; do
    # We use custom labels to suppress the time stamps which are unnecessary
    # here and would only lead to noise in version control.
    diff -u --label original --label modified ../../../tutorial/$f $f >$f.diff
done
