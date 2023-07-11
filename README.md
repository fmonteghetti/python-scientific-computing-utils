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

Install the package in editable mode:

```console
    # in a virtual environment
pip install -e /path/to/repository # virtual environment
    # in a conda environment
pip install --no-build-isolation --no-deps -e /path/to/repository
```

In the command palette of VS code:

- `File: open workspace from file` and select the desired `.code-workspace` file in the `.vscode` folder
- `Python: Select Interpreter` and select the python environment where the package has been installed.

# License

This package is distributed under the terms of the GPLv3 license.
