#!/bin/sh
# Compile dolfinx_mpc with complex PetscScalar.
# This script must be run with 'source'.
cd /tmp/dolfinx_mpc
. /usr/local/bin/dolfinx-complex-mode
. /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf
echo "Compiling dolfinx_mpc..."
rm -rf build-dir
rm -rf python/build 
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build-dir cpp/
ninja -j`nproc` install -C build-dir
pip3 install python/. --upgrade
echo "Compiling dolfinx_mpc...DONE"