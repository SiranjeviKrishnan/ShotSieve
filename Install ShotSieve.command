#!/bin/bash
# Double-click in Finder: installs ShotSieve into this folder.
cd "$(dirname "$0")" && ./install.sh --open
status=$?
echo
read -n1 -r -p "Press any key to close this window."
exit $status
