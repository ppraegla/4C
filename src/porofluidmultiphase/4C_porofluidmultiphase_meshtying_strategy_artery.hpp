// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUIDMULTIPHASE_MESHTYING_STRATEGY_ARTERY_HPP
#define FOUR_C_POROFLUIDMULTIPHASE_MESHTYING_STRATEGY_ARTERY_HPP

#include "4C_config.hpp"

#include "4C_porofluidmultiphase_meshtying_strategy_base.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace PoroMultiPhaseScaTra
{
  class PoroMultiPhaseScaTraArtCouplBase;
}

namespace POROFLUIDMULTIPHASE
{
  class MeshtyingStrategyArtery : public MeshtyingStrategyBase
  {
   public:
    //! constructor
    explicit MeshtyingStrategyArtery(POROFLUIDMULTIPHASE::TimIntImpl* porofluidmultitimint,
        const Teuchos::ParameterList& probparams, const Teuchos::ParameterList& poroparams);


    //! prepare time loop
    void prepare_time_loop() override;

    //! prepare time step
    void prepare_time_step() override;

    //! update
    void update() override;

    //! output
    void output() override;

    //! Initialize the linear solver
    void initialize_linear_solver(Teuchos::RCP<Core::LinAlg::Solver> solver) override;

    //! solve linear system of equations
    void linear_solve(Teuchos::RCP<Core::LinAlg::Solver> solver,
        Teuchos::RCP<Core::LinAlg::SparseOperator> sysmat,
        Teuchos::RCP<Core::LinAlg::Vector<double>> increment,
        Teuchos::RCP<Core::LinAlg::Vector<double>> residual,
        Core::LinAlg::SolverParams& solver_params) override;

    //! calculate norms for convergence checks
    void calculate_norms(std::vector<double>& preresnorm, std::vector<double>& incprenorm,
        std::vector<double>& prenorm,
        const Teuchos::RCP<const Core::LinAlg::Vector<double>> increment) override;

    //! create the field test
    void create_field_test() override;

    //! restart
    void read_restart(const int step) override;

    //! evaluate mesh tying
    void evaluate() override;

    //! extract increments and update mesh tying
    Teuchos::RCP<const Core::LinAlg::Vector<double>> extract_and_update_iter(
        const Teuchos::RCP<const Core::LinAlg::Vector<double>> inc) override;

    // return arterial network time integrator
    Teuchos::RCP<Adapter::ArtNet> art_net_tim_int() override { return artnettimint_; }

    //! access dof row map
    Teuchos::RCP<const Epetra_Map> artery_dof_row_map() const override;

    //! right-hand side alias the dynamic force residual for coupled system
    Teuchos::RCP<const Core::LinAlg::Vector<double>> artery_porofluid_rhs() const override;

    //! access to block system matrix of artery poro problem
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> artery_porofluid_sysmat() const override;

    //! get global (combined) increment of coupled problem
    Teuchos::RCP<const Core::LinAlg::Vector<double>> combined_increment(
        const Teuchos::RCP<const Core::LinAlg::Vector<double>> inc) const override;

    //! check if initial fields on coupled DOFs are equal
    void check_initial_fields(
        Teuchos::RCP<const Core::LinAlg::Vector<double>> vec_cont) const override;

    //! set the element pairs that are close as found by search algorithm
    void set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearbyelepairs) override;

    //! setup the strategy
    void setup() override;

    //! apply the mesh movement
    void apply_mesh_movement() const override;

    //! return blood vessel volume fraction
    Teuchos::RCP<const Core::LinAlg::Vector<double>> blood_vessel_volume_fraction() override;

   protected:
    //! artery time integration
    Teuchos::RCP<Adapter::ArtNet> artnettimint_;

    //! artery discretization
    Teuchos::RCP<Core::FE::Discretization> arterydis_;

    //! the mesh tying object
    Teuchos::RCP<PoroMultiPhaseScaTra::PoroMultiPhaseScaTraArtCouplBase> arttoporofluidcoupling_;

    //! block systemmatrix
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> comb_systemmatrix_;

    //! global rhs
    Teuchos::RCP<Core::LinAlg::Vector<double>> rhs_;

    //! global increment
    Teuchos::RCP<Core::LinAlg::Vector<double>> comb_increment_;

    //! global solution at time n+1
    Teuchos::RCP<Core::LinAlg::Vector<double>> comb_phinp_;
  };

}  // namespace POROFLUIDMULTIPHASE



FOUR_C_NAMESPACE_CLOSE

#endif
