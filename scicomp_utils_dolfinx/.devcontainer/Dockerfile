# Dockerfile providing dolfinx and dolfinx_mpc with non-root user setup. 
# Arguments:
#	TAG				https://github.com/jorgensd/dolfinx_mpc/pkgs/container/dolfinx_mpc
#	USER_NAME		name of default non-root user
#	USER_UID		UID of default non-root user
#	USER_GID		GID of default non-root user
# Build targets:
#	real	use real PetscScalar
#   complex	use complex PetscScalar

ARG TAG
FROM ghcr.io/jorgensd/dolfinx_mpc:${TAG} AS base
ARG TAG
# user defined in upstream docker
ARG USER_NAME="ubuntu"
ARG USER_UID="1000"
ARG USER_GID="1000"

# Install X virtual framebuffer and mesa (for pyvista plots)
RUN apt-get update && apt-get -y install xvfb libglx-mesa0
ENV DISPLAY=:99.0
ENV PYVISTA_OFF_SCREEN=true

# Add a passwd to root account
RUN echo 'root:root' | chpasswd

# Add a regular user with bash as default shell
# This is not needed on dolfinx v0.9+ docker
#RUN groupadd -f --gid $USER_GID $USER_NAME \
# 	&& useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME \
#	&& echo $USER_NAME:$USER_NAME | chpasswd \
#	&& chsh -s /bin/bash $USER_NAME

# Dirty fix to enable non-root user to install packages in dolfinx-env virtual
# environment
RUN chmod -R a+rwx /dolfinx-env

# Remove installed gmsh
RUN rm -rf /usr/local/lib/gmsh*
RUN rm -rf /usr/local/bin/gmsh

# Add ~/.local/bin to PATH (enables non-root pip use)
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# Target: real PetscScalar
FROM base as real

# Target: complex PetscScalar
FROM base as complex
    # New dolfinx_mpc docker image supports complex mode 
RUN echo "source /usr/local/bin/dolfinx-complex-mode" >> /etc/profile