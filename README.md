# python-scientific-computing-utils

**Table of Contents**

- [Description](#description)
- [Installation](#installation)
- [License](#license)

Collection of elementary wrappers (functions and classes) for rapid prototyping of numerical methods for partial differential equations in python. Tutorial scripts demonstrating the modules are available in the `demos` folder.

The provided modules rely on the following packages:
- `gmsh`: mesh generation.
- `fenics` or `fenicsx`: finite element assembly.
- `petsc4py`: sparse linear algebra.
- `slepc`: sparse eigensolver.

## Installation

Build and install with

```console
hatch build
pip install dist/scientific_computing_utils-*.whl
```

To modify the package, perform a local editable installation with

```console
pip install -e /path/to/repository
```

When working within a conda environment, append the arguments `--no-build-isolation --no-deps` to the above `pip` commands.

## License

`scientific-computing-utils` is distributed under the terms of the GPLv3 license.
