// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_DYN_HPP
#define FOUR_C_FSI_DYN_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN

void fluid_ale_drt();
void fluid_xfem_drt();
void fluid_fluid_fsi_drt();

/*! \brief Entry routine to all ALE-based FSI algorithms
 *
 *  All ALE-based FSI algorithms rely on a ceratin DOF ordering, namely
 *
 *               structure dof < fluid dof < ale dof
 *
 *  We establish this ordering by calling fill_complete() on the three
 *  discretizations in the order (1) structure (2) fluid (3) ALE.
 */
void fsi_ale_drt();
void xfsi_drt();
void xfpsi_drt();

void fsi_immersed_drt();

FOUR_C_NAMESPACE_CLOSE

#endif
