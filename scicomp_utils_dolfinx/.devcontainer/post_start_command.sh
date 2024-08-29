#!/bin/sh
# Setup docker environment.
containerWorkspace="$1"
# Launch X virtual framebuffer in the background (neede for pyvista)
# (Check with ps -ef | grep Xvfb; the 'sleep 3' is important.)
nohup Xvfb :99 -screen 0 800x600x24 &
sleep 3