#!/bin/sh
# Setup docker environment.
containerWorkspace="$1"
# Upgrade pip (old versions to do not support local editable installs)
pip3 install --upgrade pip
# Needed for gmsh
mkdir -p ~/.local/bin
ln -s /usr/bin/python3 ~/.local/bin/python 
# Local editable install
pip install -e ${containerWorkspace}
# For vscode jupyter extension
pip install ipykernel pandas