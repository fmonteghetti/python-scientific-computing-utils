# scicomp_utils_dolfin

**Table of Contents**

- [Description](#description)
- [Installation](#installation)
- [Development](#development)
- [License](#license)

# Description

Collection of elementary wrappers (functions and classes) for rapid prototyping of numerical methods for partial differential equations in python using `fenics` and `multiphenics`. Tutorial scripts demonstrating the modules are available in the `demos` folder.

The provided modules rely on the following packages:
- `gmsh`: mesh generation.
- `fenics` and `multiphenics`: finite element assembly.
- `petsc4py`: sparse linear algebra.
- `slepc`: sparse eigensolver.

# Installation

```console
pip install . 
```

# Development

Install the package in editable mode:

```console
    # in a virtual environment
pip install -e . 
    # in a conda environment
pip install --no-build-isolation --no-deps -e . 
```

# License

This package is distributed under the terms of the GPLv3 license.
