// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_so3_hex8.hpp"

FOUR_C_NAMESPACE_OPEN


void Discret::Elements::SoHex8::soh8_element_center_refe_coords(
    Core::LinAlg::Matrix<NUMDIM_SOH8, 1>& centercoord,
    Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> const& xrefe) const
{
  Core::LinAlg::Matrix<NUMNOD_SOH8, 1> funct;
  Core::FE::shape_function_3d(funct, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);
  centercoord.multiply_tn(xrefe, funct);
  return;
}


void Discret::Elements::SoHex8::soh8_gauss_point_refe_coords(
    Core::LinAlg::Matrix<NUMDIM_SOH8, 1>& gpcoord,
    Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> const& xrefe, int const gp) const
{
  const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts = soh8_shapefcts();
  Core::LinAlg::Matrix<NUMNOD_SOH8, 1> funct(true);
  funct = shapefcts[gp];
  gpcoord.multiply_tn(xrefe, funct);

  return;
}

FOUR_C_NAMESPACE_CLOSE
