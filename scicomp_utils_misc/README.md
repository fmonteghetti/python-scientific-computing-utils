# scicomp_utils_mesh

**Table of Contents**

- [Description](#description)
- [Installation](#installation)
- [Development](#development)
- [License](#license)

# Description

Collection of miscellaneous utility functions. 

# Installation

```console
pip install "scicomp_utils_misc @ git+https://github.com/fmonteghetti/python-scientific-computing-utils/@branch#subdirectory=scicomp_utils_misc"
```

# Development

## Local

Install the package in editable mode:

```console
    # in a virtual environment
pip install -e "scicomp_utils_misc @ git+https://github.com/fmonteghetti/python-scientific-computing-utils@branch#subdirectory=scicomp_utils_misc"
    # in a conda environment
pip install --no-build-isolation --no-deps -e "scicomp_utils_misc @ git+https://github.com/fmonteghetti/python-scientific-computing-utils@branch#subdirectory=scicomp_utils_misc"
```

## Docker via devcontainers (Visual Studio Code)

The file `.devcontainer/devcontainer.json` provides a development container.

To use it, in the command palette of VS code:

- `Dev Containers: Open Folder in Container` and select root of this repository.
- `Python: Select Interpreter` and select the python environment.

# License

This package is distributed under the terms of the GPLv3 license.