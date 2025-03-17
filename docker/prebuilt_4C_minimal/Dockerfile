# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# TODO make a new image instead of removing stuff from the dependencies image

FROM ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:latest

USER root

# Remove unnecessary packages
RUN apt-get update && apt-get -y remove \
      doxygen \
      clang* \
      llvm* \
  && apt-get -y autoremove \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /usr/share/doc

# Remove testing dependencies
RUN rm -r /opt/4C-dependencies-testing/

RUN git clone --depth 1 https://github.com/4C-multiphysics/4C.git 4C-source

# TODO remove hack when the preset is on main
COPY CMakePresets.json 4C-source/presets/docker

RUN mkdir 4C-source/build && cd 4C-source/build && \
  cmake ../ --fresh -DCMAKE_INSTALL_PREFIX=/home/user/4C --preset=docker_minimal && \
  cmake --build . --target install -- -j `nproc`

# Remove build artifacts
RUN rm -r /home/user/4C-source

# Copy everything to a new image to get rid of the previous layers
FROM ubuntu:24.04
COPY --from=0 / /
LABEL org.opencontainers.image.description="Image containing the 4C executable"
LABEL org.4c-multiphysics.project=4C

USER user
WORKDIR /home/user
