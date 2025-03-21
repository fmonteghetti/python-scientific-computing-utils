# scicomp_utils_mesh

# Description

Collection of elementary wrappers around `gmsh` and `meshio`. Tutorial scripts demonstrating the modules are available in the `demos` folder.

# Installation

## Local installation

The package can be installed in a new conda environment with:

```console
conda env create --file environment.yml 
```

For development, install in editable mode with:

```console
conda env create --file environment_dev.yml 
```

## Containerized installation (docker via devcontainers)

The file `.devcontainer/devcontainer.json` provides a development container.

To use it, in the command palette of VS code:

- `Dev Containers: Open Folder in Container` and select root of this repository.

# Development

Format code with
```
ruff format
```

# License

This package is distributed under the terms of the GPLv3 license.