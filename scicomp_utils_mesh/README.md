# scicomp_utils_mesh

**Table of Contents**

- [Description](#description)
- [Installation](#installation)
- [Development](#development)
- [License](#license)

# Description

Collection of elementary wrappers around `gmsh` and `meshio`. Tutorial scripts demonstrating the modules are available in the `demos` folder.

# Installation

```console
pip install . 
```

# Development

## Local

Install the package in editable mode:

```console
    # in a virtual environment
pip install -e . 
    # in a conda environment
pip install --no-build-isolation --no-deps -e . 
```

## Docker via devcontainers (Visual Studio Code)

The file `.devcontainer/devcontainer.json` provides a development container.

To use it, in the command palette of VS code:

- `Dev Containers: Open Folder in Container` and select root of this repository.
- `Python: Select Interpreter` and select the python environment.

# License

This package is distributed under the terms of the GPLv3 license.