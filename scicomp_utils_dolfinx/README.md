# python-scientific-computing-utils

**Table of Contents**

- [Description](#description)
- [Installation](#installation)
- [Development in Visual Studio Code](#development-in-visual-studio-code)
- [License](#license)

# Description

Collection of elementary wrappers (functions and classes) for rapid prototyping of numerical methods for partial differential equations in python. Tutorial scripts demonstrating the modules are available in the `demos` folder.

The provided modules rely on the following packages:
- `gmsh`: mesh generation.
- `fenics` or `fenicsx`: finite element assembly.
- `petsc4py`: sparse linear algebra.
- `slepc`: sparse eigensolver.

# Installation

Build and install with:

```console
hatch build
pip install dist/scientific_computing_utils-*.whl
```

# Development in Visual Studio Code

## Local

Install the package in editable mode:

```console
    # in a virtual environment
pip install -e /path/to/repository # virtual environment
    # in a conda environment
pip install --no-build-isolation --no-deps -e /path/to/repository
```

In the command palette of VS code:

- `File: Open Workspace from File` and select the desired `.code-workspace` file in the `.vscode` folder.
- `Python: Select Interpreter` and select the python environment where the package has been installed.

## Docker (for dolfinx)

The file `.devcontainer/devcontainer.json` provides a development container that contains the `dolfinx` and `dolfinx_mpc` packages.

To use it, in the command palette of VS code:

- `Dev Containers: Open Folder in Container` and select root of this repository.
- `File: Open Workspace from File` and select `.vscode/docker.code-workspace`.
- `Python: Select Interpreter` and select the python environment.

You can switch between real and complex `PetscScalar` by editing the `target` field in `.devcontainer/devcontainer.json` and selecting `Dev Containers: Rebuild Container` in the VS code command palette.

# License

This package is distributed under the terms of the GPLv3 license.
