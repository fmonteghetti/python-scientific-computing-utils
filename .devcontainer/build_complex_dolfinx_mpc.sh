#!/bin/sh
# Compile dolfinx_mpc with complex PetscScalar.
# This script must be run with 'source'.
echo "Switching dolfinx to complex mode..."
. /usr/local/bin/dolfinx-complex-mode
echo "Compiling dolfinx_mpc..."
cd /tmp/dolfinx_mpc/cpp
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir ../cpp/
ninja -j3 install -C build-dir
cd .. && pip3 install python/. --upgrade
echo "Compiling dolfinx_mpc...DONE"