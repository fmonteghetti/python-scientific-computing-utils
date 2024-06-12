#!/bin/sh
# Setup docker environment.
containerWorkspace="$1"
# Local editable install
pip install -e ${containerWorkspace}

