# scicomp_utils_dolfin

# Description

Collection of elementary wrappers (functions and classes) for rapid prototyping of numerical methods for partial differential equations in python using `fenics` and `multiphenics`. Tutorial scripts demonstrating the modules are available in the `demos` folder.

The provided modules rely on the following packages:
- `gmsh`: mesh generation.
- `fenics` and `multiphenics`: finite element assembly.
- `petsc4py`: sparse linear algebra.
- `slepc`: sparse eigensolver.

# Installation

The package `scicomp_utils_dolfin` can be installed in a new conda environment with:

```console
conda env create --file environment.yml 
```

For development, install in editable mode with

```console
conda env create --file environment_dev.yml 
```

# License

This package is distributed under the terms of the GPLv3 license.
