# Dockerfile providing python and gmsh.  

ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION} AS base

ARG USER_NAME
ARG USER_UID
ARG USER_GID
ARG GMSH_VERSION

# Install gmsh dependencies.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    libglu1 \
    libxcursor-dev \
    libxft2 \
    libxinerama1 \
    libfltk1.3-dev \
    libfreetype6-dev  \
    libgl1-mesa-dev \
    libocct-foundation-dev \
    libocct-data-exchange-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

    # Install:
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
