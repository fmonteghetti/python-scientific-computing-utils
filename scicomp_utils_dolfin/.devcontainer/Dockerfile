# Dockerfile providing dolfinx and dolfinx_mpc. 
# Arguments:
#	TAG				https://hub.docker.com/r/dokken92/dolfinx_mpc/tags	
#	USER_NAME		name of default non-root user
#	USER_UID		UID of default non-root user
#	USER_GID		GID of default non-root user
#	PYVISTA_VERSION	version of pyvista
#	GMSH_VERSION	version of gmsh
# Build targets:
#	real	use real PetscScalar
#   complex	use complex PetscScalar
ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION} AS base

ARG USER_NAME
ARG USER_UID
ARG USER_GID
ARG GMSH_VERSION

# Install:
#   - pyvista
#   - ipykernel (interactive window in vscode)
#   - Pandas (array inspections in vs code)
RUN pip3 install --upgrade --no-cache-dir \
                 ipykernel \
                 gmsh==${GMSH_VERSION} matplotlib \
                 pandas

# Add a passwd to root account
RUN echo 'root:root' | chpasswd

# Add a regular user with bash as default shell
RUN groupadd --gid $USER_GID $USER_NAME \
 	&& useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME \
	&& echo $USER_NAME:$USER_NAME | chpasswd \
	&& chsh -s /bin/bash $USER_NAME
