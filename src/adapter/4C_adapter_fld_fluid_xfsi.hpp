// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_FLD_FLUID_XFSI_HPP
#define FOUR_C_ADAPTER_FLD_FLUID_XFSI_HPP

#include "4C_config.hpp"

#include "4C_adapter_fld_wrapper.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations

namespace Core::LinAlg
{
  class Solver;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace FLD
{
  class XFluid;
  namespace Utils
  {
    class MapExtractor;
  }
}  // namespace FLD

namespace XFEM
{
  class MeshCouplingFSI;
}

namespace Adapter
{
  class XFluidFSI : public FluidWrapper
  {
   public:
    /// Constructor
    XFluidFSI(Teuchos::RCP<Fluid> fluid,
        const std::string coupling_name,  // name of the FSI coupling condition
        Teuchos::RCP<Core::LinAlg::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
        Teuchos::RCP<Core::IO::DiscretizationWriter> output);

    /// initialize algorithm
    void init() override;

    /// communication object at the interface
    virtual Teuchos::RCP<FLD::Utils::MapExtractor> const& struct_interface() const
    {
      return structinterface_;
    }

    /// communication object at the interface
    Teuchos::RCP<FLD::Utils::MapExtractor> const& interface() const override { return interface_; }

    /// communication object at the interface without pressure dofs for FPSI problems
    Teuchos::RCP<FLD::Utils::MapExtractor> const& fpsi_interface() const override
    {
      return fpsiinterface_;
    }

    /// Velocity-displacement conversion at the fsi interface
    double time_scaling() const override;

    /// Return interface forces
    virtual Teuchos::RCP<Core::LinAlg::Vector<double>> extract_struct_interface_forces();

    /// Return interface velocity at old time level n
    virtual Teuchos::RCP<Core::LinAlg::Vector<double>> extract_struct_interface_veln();

    /// Return interface velocity at new time level n+1
    virtual Teuchos::RCP<Core::LinAlg::Vector<double>> extract_struct_interface_velnp()
    {
      FOUR_C_THROW("Not implemented, yet!");
      return Teuchos::null;
    }

    /// apply the interface velocities to the fluid
    virtual void apply_struct_interface_velocities(Teuchos::RCP<Core::LinAlg::Vector<double>> ivel);

    /// apply the interface displacements to the fluid
    virtual void apply_struct_mesh_displacement(
        Teuchos::RCP<const Core::LinAlg::Vector<double>> interface_disp);

    /// convert increment of displacement to increment in velocity
    void displacement_to_velocity(Teuchos::RCP<Core::LinAlg::Vector<double>> fcx) override;

    /// Apply initial mesh displacement
    void apply_initial_mesh_displacement(
        Teuchos::RCP<const Core::LinAlg::Vector<double>> initfluiddisp) override
    {
      FOUR_C_THROW("Not implemented, yet!");
    }

    /// apply the interface displacements to the fluid
    void apply_mesh_displacement(
        Teuchos::RCP<const Core::LinAlg::Vector<double>> fluiddisp) override;

    void set_mesh_map(Teuchos::RCP<const Epetra_Map> mm, const int nds_master = 0) override;

    /// return coupling matrix between fluid and structure as sparse matrices
    Teuchos::RCP<Core::LinAlg::SparseMatrix> c_struct_fluid_matrix();
    Teuchos::RCP<Core::LinAlg::SparseMatrix> c_fluid_struct_matrix();
    Teuchos::RCP<Core::LinAlg::SparseMatrix> c_struct_struct_matrix();

    Teuchos::RCP<const Core::LinAlg::Vector<double>> rhs_struct_vec();

    Teuchos::RCP<FLD::XFluid> my_fluid() { return xfluid_; }

    /// return boundary discretization
    Teuchos::RCP<Core::FE::Discretization> boundary_discretization();

    bool newton_restart_monolithic() { return xfluid_->newton_restart_monolithic(); }

    Teuchos::RCP<std::map<int, int>> get_permutation_map()
    {
      return xfluid_->get_permutation_map();
    }

    /// GmshOutput for background mesh and cut mesh
    void gmsh_output(const std::string& name,  ///< name for output file
        const int step,                        ///< step number
        const int count,                       ///< counter for iterations within a global time step
        Core::LinAlg::Vector<double>& vel,     ///< vector holding velocity and pressure dofs
        Teuchos::RCP<Core::LinAlg::Vector<double>> acc =
            Teuchos::null  ///< vector holding accelerations
    );

   protected:
    /// A casted pointer to the fluid itself
    Teuchos::RCP<FLD::XFluid> xfluid_;

    /// the interface map setup for fsi interface, interior translation
    Teuchos::RCP<FLD::Utils::MapExtractor> interface_;

    /// the interface map setup for fsi interface, interior translation
    Teuchos::RCP<FLD::Utils::MapExtractor> structinterface_;

    /// the interface map setup for fpsi interface
    Teuchos::RCP<FLD::Utils::MapExtractor> fpsiinterface_;

    /// ALE dof map
    Teuchos::RCP<Core::LinAlg::MapExtractor> meshmap_;
    Teuchos::RCP<Epetra_Map> permfluidmap_;
    Teuchos::RCP<Epetra_Map> fullfluidmap_;

    //! @name local copies of input parameters
    std::string coupling_name_;  /// the name of the XFEM::MeshCoupling object
    Teuchos::RCP<XFEM::MeshCouplingFSI> mesh_coupling_fsi_;
    Teuchos::RCP<Core::LinAlg::Solver> solver_;
    Teuchos::RCP<Teuchos::ParameterList> params_;
    //@}
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
