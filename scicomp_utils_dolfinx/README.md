# scicomp_utils_dolfinx

# Description

Collection of elementary wrappers (functions and classes) for rapid prototyping of numerical methods for partial differential equations using `dolfinx`. Tutorial scripts demonstrating the modules are available in the `demos` folder.

The provided modules rely on the following packages:
- `gmsh`: mesh generation.
- `dolfinx`: finite element assembly.
- `petsc4py`: sparse linear algebra.
- `slepc`: sparse eigensolver.

# Installation

## Local installation

The package `scicomp_utils_dolfinx` can be installed in a new conda environment with:

```console
conda env create --file environment.yml 
```

For development, install `scicomp_utils_dolfinx` in editable mode with

```console
conda env create --file environment_dev.yml 
```

## Containerized installation (docker via devcontainers)

The file `.devcontainer/devcontainer.json` provides a development container that contains the `dolfinx_mpc` package.

To use it, in the command palette of VS code:

- `Dev Containers: Open Folder in Container` and select root of this repository.
- `Python: Select Interpreter` and select the python environment.

You can switch between real and complex `PetscScalar` by editing the `target` field in `.devcontainer/devcontainer.json` and selecting `Dev Containers: Rebuild Container` in the VS code command palette.

# License

This package is distributed under the terms of the GPLv3 license.