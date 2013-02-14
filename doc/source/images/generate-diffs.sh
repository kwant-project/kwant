# !/bin/sh

# This script regenerates the .diff files in this directory.  It's these files
# that are kept under vesion control instead of the scripts themselves.

for f in [a-zA-Z]*.py; do
    echo $f
    # We use custom labels to suppress the time stamps which are unnecessary
    # here and would only lead to noise in version control.
    grep -v '#HIDDEN' ../tutorial/$f |
    diff -u --label original --label modified - $f >$f.diff
done
