# scicomp_utils_dolfin

**Table of Contents**

- [Description](#description)
- [Installation](#installation)
- [Development](#development)
- [License](#license)

# Description

Collection of elementary wrappers (functions and classes) for rapid prototyping of numerical methods for partial differential equations in python. Tutorial scripts demonstrating the modules are available in the `demos` folder.

The provided modules rely on the following packages:
- `gmsh`: mesh generation.
- `fenics`: finite element assembly.
- `petsc4py`: sparse linear algebra.
- `slepc`: sparse eigensolver.

# Installation

```console
pip install "scicomp_utils_dolfinx @ git+https://github.com/fmonteghetti/python-scientific-computing-utils/@branch#subdirectory=scicomp_utils_dolfinx"
```

# Development

Install the package in editable mode:

```console
    # in a virtual environment
pip install -e "scicomp_utils_dolfin @ git+https://github.com/fmonteghetti/python-scientific-computing-utils@branch#subdirectory=scicomp_utils_dolfin"
    # in a conda environment
pip install --no-build-isolation --no-deps -e "scicomp_utils_dolfin @ git+https://github.com/fmonteghetti/python-scientific-computing-utils@branch#subdirectory=scicomp_utils_dolfin"
```

# License

This package is distributed under the terms of the GPLv3 license.
