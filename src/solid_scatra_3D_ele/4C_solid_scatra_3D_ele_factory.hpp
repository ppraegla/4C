/*! \file

\brief Factory of solid-scatra elements

\level 1
*/

#ifndef FOUR_C_SOLID_SCATRA_3D_ELE_FACTORY_HPP
#define FOUR_C_SOLID_SCATRA_3D_ELE_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_discretization_fem_general_cell_type_traits.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_solid_3D_ele_calc_displacement_based.hpp"
#include "4C_solid_3D_ele_factory_lib.hpp"
#include "4C_solid_scatra_3D_ele_calc.hpp"

FOUR_C_NAMESPACE_OPEN

namespace DRT::ELEMENTS
{
  /*!
   *  @brief struct for managing solidscatra element properties
   */
  struct SolidScatraElementProperties
  {
    //! scalar transport implementation type (physics)
    INPAR::SCATRA::ImplType impltype{INPAR::SCATRA::ImplType::impltype_undefined};
  };

  namespace DETAILS
  {

    using ImplementedSolidScatraCellTypes = CORE::FE::CelltypeSequence<CORE::FE::CellType::hex8,
        CORE::FE::CellType::hex27, CORE::FE::CellType::tet4, CORE::FE::CellType::tet10>;

    template <CORE::FE::CellType celltype>
    using DisplacementBasedSolidScatraIntegrator =
        SolidScatraEleCalc<celltype, DisplacementBasedFormulation<celltype>>;

    using DisplacementBasedSolidScatraEvaluator =
        CORE::FE::apply_celltype_sequence<DisplacementBasedSolidScatraIntegrator,
            ImplementedSolidScatraCellTypes>;

    using SolidScatraEvaluators = CORE::FE::Join<DisplacementBasedSolidScatraEvaluator>;
  }  // namespace DETAILS

  /// Variant holding the different implementations for the solid-scatra element
  using SolidScatraCalcVariant = CreateVariantType<DETAILS::SolidScatraEvaluators>;

  SolidScatraCalcVariant CreateSolidScatraCalculationInterface(CORE::FE::CellType celltype);

  template <CORE::FE::CellType celltype>
  SolidScatraCalcVariant CreateSolidScatraCalculationInterface();
}  // namespace DRT::ELEMENTS

FOUR_C_NAMESPACE_CLOSE

#endif