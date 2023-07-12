#!/bin/sh
# Setup docker environment.
containerWorkspace="$1"
# Local editable install of scientific-computing-utils package
pip install -e ${containerWorkspace}
