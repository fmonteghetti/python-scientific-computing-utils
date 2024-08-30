#!/bin/sh
# Setup docker environment.
containerWorkspace="$1"
# Upgrade pip (old versions to do not support local editable installs)
pip3 install --upgrade pip
# Needed for gmsh
ln -s /usr/bin/python3 ~/.local/bin/python 
# Local editable install
pip3 install -e ${containerWorkspace}
# ipykernel (interactive window in vscode)
# pandas (array inspections in vs code)
pip3 install ipykernel pandas matplotlib