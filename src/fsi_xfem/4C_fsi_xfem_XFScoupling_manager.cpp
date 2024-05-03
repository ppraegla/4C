/*----------------------------------------------------------------------*/
/*! \file
\brief  Coupling Manager for eXtended Fluid Structural Coupling

\level 2


*----------------------------------------------------------------------*/
#include "4C_fsi_xfem_XFScoupling_manager.hpp"

#include "4C_adapter_str_structure.hpp"
#include "4C_fluid_xfluid.hpp"
#include "4C_fluid_xfluid_fluid.hpp"  //Todo: remove me finally
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_xfem_condition_manager.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------*
| Constructor                                                                 ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
XFEM::XfsCouplingManager::XfsCouplingManager(Teuchos::RCP<ConditionManager> condmanager,
    Teuchos::RCP<ADAPTER::Structure> structure, Teuchos::RCP<FLD::XFluid> xfluid,
    std::vector<int> idx)
    : CouplingCommManager(structure->Discretization(), "XFEMSurfFSIMono", 0, 3),
      struct_(structure),
      xfluid_(xfluid),
      cond_name_("XFEMSurfFSIMono"),
      idx_(idx),
      interface_second_order_(false)
{
  if (idx_.size() != 2)
    FOUR_C_THROW("XFSCoupling_Manager required two block ( 2 != %d)", idx_.size());

  const Teuchos::ParameterList& fsidyn = GLOBAL::Problem::Instance()->FSIDynamicParams();
  interface_second_order_ = CORE::UTILS::IntegralValue<int>(fsidyn, "SECONDORDER");

  // Coupling_Comm_Manager create all Coupling Objects now with Structure has idx = 0, Fluid has idx
  // = 1!
  mcfsi_ =
      Teuchos::rcp_dynamic_cast<XFEM::MeshCouplingFSI>(condmanager->GetMeshCoupling(cond_name_));
  if (mcfsi_ == Teuchos::null) FOUR_C_THROW(" Failed to get MeshCouplingFSI for Structure!");

  mcfsi_->SetTimeFac(1. / GetInterfaceTimefac());

  // safety check
  if (!mcfsi_->IDispnp()->Map().SameAs(*GetMapExtractor(0)->Map(1)))
    FOUR_C_THROW("XFSCoupling_Manager: Maps of Condition and Mesh Coupling do not fit!");

  // storage of the resulting Robin-type structural forces from the old timestep
  // Recovering of Lagrange multiplier happens on fluid field
  lambda_ = Teuchos::rcp(new Epetra_Vector(*mcfsi_->GetCouplingDis()->DofRowMap(), true));
}

/*-----------------------------------------------------------------------------------------*
| Set required displacement & velocity states in the coupling object          ager 04/2017 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::InitCouplingStates()
{
  // 1 Set Displacement on both mesh couplings ... we get them from the structure field!
  InsertVector(0, struct_->Dispn(), 0, mcfsi_->IDispn(), CouplingCommManager::full_to_partial);
  InsertVector(0, struct_->Dispn(), 0, mcfsi_->IDispnp(), CouplingCommManager::full_to_partial);

  // 2 Set Displacement on both mesh couplings ... we get them from the structure field!
  InsertVector(0, struct_->Veln(), 0, mcfsi_->IVeln(), CouplingCommManager::full_to_partial);
  InsertVector(0, struct_->Veln(), 0, mcfsi_->IVelnp(), CouplingCommManager::full_to_partial);
}

/*-----------------------------------------------------------------------------------------*
| Set required displacement & velocity states in the coupling object          ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::SetCouplingStates()
{
  // 1 update last increment, before we set new idispnp
  mcfsi_->UpdateDisplacementIterationVectors();

  // 2 Set Displacement on both mesh couplings ... we get them from the structure field!
  InsertVector(0, struct_->Dispnp(), 0, mcfsi_->IDispnp(), CouplingCommManager::full_to_partial);

  // get interface velocity at t(n)
  Teuchos::RCP<Epetra_Vector> velnp =
      Teuchos::rcp(new Epetra_Vector(mcfsi_->IVelnp()->Map(), true));
  velnp->Update(1.0, *mcfsi_->IDispnp(), -1.0, *mcfsi_->IDispn(), 0.0);

  // inverse of FSI (1st order, 2nd order) scaling
  const double scaling_FSI = GetInterfaceTimefac();  // 1/(theta_FSI * dt) =  1/weight^FSI_np
  const double dt = xfluid_->Dt();

  // v^{n+1} = -(1-theta)/theta * v^{n} - 1/(theta*dt)*(d^{n+1}-d^{n0})
  velnp->Update(-(dt - 1 / scaling_FSI) * scaling_FSI, *mcfsi_->IVeln(), scaling_FSI);

  // 3 Set Structural Velocity onto ps mesh coupling
  InsertVector(0, velnp, 0, mcfsi_->IVelnp(), CouplingCommManager::partial_to_partial);

  // 4 Set Structural Velocity onto the structural discretization
  if (mcfsi_->GetAveragingStrategy() != INPAR::XFEM::Xfluid_Sided)
  {
    // Set Dispnp (used to calc local coord of gausspoints)
    struct_->Discretization()->SetState("dispnp", struct_->Dispnp());
    // Set Velnp (used for interface integration)
    Teuchos::RCP<Epetra_Vector> fullvelnp =
        Teuchos::rcp(new Epetra_Vector(struct_->Velnp()->Map(), true));
    fullvelnp->Update(1.0, *struct_->Dispnp(), -1.0, *struct_->Dispn(), 0.0);
    fullvelnp->Update(-(dt - 1 / scaling_FSI) * scaling_FSI, *struct_->Veln(), scaling_FSI);
    struct_->Discretization()->SetState("velaf", fullvelnp);
  }
}

/*-----------------------------------------------------------------------------------------*
| Add the coupling matrixes to the global systemmatrix                        ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::AddCouplingMatrix(
    CORE::LINALG::BlockSparseMatrixBase& systemmatrix, double scaling)
{
  /*----------------------------------------------------------------------*/
  // Coupling blocks C_sf, C_fs and C_ss
  /*----------------------------------------------------------------------*/
  CORE::LINALG::SparseMatrix& C_ss_block = (systemmatrix)(idx_[0], idx_[0]);
  /*----------------------------------------------------------------------*/
  // scaling factor for displacement <-> velocity conversion (FSI)
  // inverse of FSI (1st order, 2nd order) scaling
  const double scaling_FSI = GetInterfaceTimefac();  // 1/(theta_FSI * dt) =  1/weight^FSI_np

  // * all the coupling matrices are scaled with the weighting of the fluid w.r.t new time step np
  //    -> Unscale the blocks with (1/(theta_f*dt) = 1/weight(t^f_np))
  // * additionally the C_*s blocks (C_ss and C_fs) have to include the conversion from structural
  // displacements to structural velocities
  //    -> Scale these blocks with (1/(theta_FSI*dt) = 1/weight(t^FSI_np))
  //
  // REMARK that Scale() scales the original coupling matrix in xfluid

  // C_ss_block scaled with 1/(theta_f*dt) * 1/(theta_FSI*dt) = 1/weight(t^f_np) *
  // 1/weight(t^FSI_np) add the coupling block C_ss on the already existing diagonal block
  C_ss_block.Add(*xfluid_->C_ss_Matrix(cond_name_), false, scaling * scaling_FSI, 1.0);


  GLOBAL::ProblemType probtype = GLOBAL::Problem::Instance()->GetProblemType();

  // Todo: Need to eighter split fluid matrixes in the fsi algo or change the maps of the coupling
  // matrixes(merged)
  bool is_xfluidfluid =
      Teuchos::rcp_dynamic_cast<FLD::XFluidFluid>(xfluid_, false) != Teuchos::null;

  if (probtype == GLOBAL::ProblemType::fsi_xfem &&
      !is_xfluidfluid)  // use assign for off diagonal blocks
  {
    // scale the off diagonal coupling blocks
    xfluid_->C_sx_Matrix(cond_name_)
        ->Scale(scaling);  //<   1/(theta_f*dt)                    = 1/weight(t^f_np)
    xfluid_->C_xs_Matrix(cond_name_)
        ->Scale(scaling * scaling_FSI);  //<   1/(theta_f*dt) * 1/(theta_FSI*dt) = 1/weight(t^f_np)
                                         //* 1/weight(t^FSI_np)

    systemmatrix.Assign(idx_[0], idx_[1], CORE::LINALG::View, *xfluid_->C_sx_Matrix(cond_name_));
    systemmatrix.Assign(idx_[1], idx_[0], CORE::LINALG::View, *xfluid_->C_xs_Matrix(cond_name_));
  }
  else if (probtype == GLOBAL::ProblemType::fpsi_xfem || is_xfluidfluid)
  {
    CORE::LINALG::SparseMatrix& C_fs_block = (systemmatrix)(idx_[1], idx_[0]);
    CORE::LINALG::SparseMatrix& C_sf_block = (systemmatrix)(idx_[0], idx_[1]);

    C_sf_block.Add(*xfluid_->C_sx_Matrix(cond_name_), false, scaling, 1.0);
    C_fs_block.Add(*xfluid_->C_xs_Matrix(cond_name_), false, scaling * scaling_FSI, 1.0);
  }
  else
  {
    FOUR_C_THROW("XFSCoupling_Manager: Want to use me for other problemtype --> check and add me!");
  }
}

/*-----------------------------------------------------------------------------------------*
| Add the coupling rhs                                                        ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::AddCouplingRHS(
    Teuchos::RCP<Epetra_Vector> rhs, const CORE::LINALG::MultiMapExtractor& me, double scaling)
{
  Teuchos::RCP<Epetra_Vector> coup_rhs_sum = Teuchos::rcp(new Epetra_Vector(*xfluid_->RHS_s_Vec(
      cond_name_)));  // REMARK: Copy this vector to store the correct lambda_ in update!
  /// Lagrange multiplier \lambda_\Gamma^n at the interface (ie forces onto the structure,
  /// Robin-type forces consisting of fluid forces and the Nitsche penalty term contribution)
  if (lambda_ != Teuchos::null)
  {
    /*----------------------------------------------------------------------*/
    // get time integration parameters of structure and fluid time integrators
    // to enable consistent time integration among the fields
    /*----------------------------------------------------------------------*/

    /*----------------------------------------------------------------------*/
    // this is the interpolation weight for quantities from last time step
    // alpha_f for genalpha and (1-theta) for OST (weighting of the old time step n for
    // displacements)
    const double stiparam = struct_->TimIntParam();  // (1-theta) for OST and alpha_f for Genalpha

    // scale factor for the structure system matrix w.r.t the new time step
    const double scaling_S = 1.0 / (1.0 - stiparam);  // 1/(1-alpha_F) = 1/weight^S_np
    // add Lagrange multiplier (structural forces from t^n)
    int err = coup_rhs_sum->Update(stiparam * scaling_S, *lambda_, scaling);
    if (err) FOUR_C_THROW("Update of Nit_Struct_FSI RHS failed with errcode = %d!", err);
  }
  else
  {
    coup_rhs_sum->Scale(scaling);
  }

  Teuchos::RCP<Epetra_Vector> coup_rhs = Teuchos::rcp(new Epetra_Vector(*me.Map(idx_[0]), true));
  CORE::LINALG::Export(*coup_rhs_sum, *coup_rhs);  // use this command as long as poro ist not split
                                                   // into two bocks in the monolithic algorithm!
  // InsertVector(0,coup_rhs_sum,0,coup_rhs,Coupling_Comm_Manager::partial_to_full);
  me.AddVector(coup_rhs, idx_[0], rhs);
}

/*----------------------------------------------------------------------*/
/* Store the Coupling RHS of the Old Timestep in lambda     ager 06/2016 |
 *----------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::Update(double scaling)
{
  /*----------------------------------------------------------------------*/
  // we directly store the fluid-unscaled rhs_C_s residual contribution from the fluid solver which
  // corresponds to the actual acting forces

  // scaling for the structural residual is done when it is added to the global residual vector
  // get the coupling rhs from the xfluid, this vector is based on the boundary dis which is part of
  // the structure dis
  lambda_->Update(scaling, *xfluid_->RHS_s_Vec(cond_name_), 0.0);
  return;
}

/*----------------------------------------------------------------------*/
/* Write Output                                             ager 06/2016 |
 *-----------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::Output(IO::DiscretizationWriter& writer)
{
  //--------------------------------
  // output for Lagrange multiplier field (ie forces onto the structure, Robin-type forces
  // consisting of fluid forces and the Nitsche penalty term contribution)
  //--------------------------------
  Teuchos::RCP<Epetra_Vector> lambdafull =
      Teuchos::rcp(new Epetra_Vector(*GetMapExtractor(0)->FullMap(), true));
  InsertVector(0, lambda_, 0, lambdafull, CouplingCommManager::partial_to_full);
  writer.WriteVector("fsilambda", lambdafull);
  return;
}
/*----------------------------------------------------------------------*/
/* Read Restart on the interface                            ager 06/2016 |
 *-----------------------------------------------------------------------*/
void XFEM::XfsCouplingManager::ReadRestart(IO::DiscretizationReader& reader)
{
  Teuchos::RCP<Epetra_Vector> lambdafull =
      Teuchos::rcp(new Epetra_Vector(*GetMapExtractor(0)->FullMap(), true));
  reader.ReadVector(lambdafull, "fsilambda");
  InsertVector(0, lambdafull, 0, lambda_, CouplingCommManager::full_to_partial);
  return;
}

/*-----------------------------------------------------------------------------------------*
| Get Timeface on the interface (for OST this is 1/(theta dt))                ager 06/2016 |
*-----------------------------------------------------------------------------------------*/
double XFEM::XfsCouplingManager::GetInterfaceTimefac()
{
  /*
   * Delta u(n+1,i+1) = fac * (Delta d(n+1,i+1) - dt * u(n))
   *
   *             / = 2 / dt   if interface time integration is second order
   * with fac = |
   *             \ = 1 / dt   if interface time integration is first order
   */
  const double dt = xfluid_->Dt();
  if (interface_second_order_)
    return 2. / dt;
  else
    return 1. / dt;
}

FOUR_C_NAMESPACE_CLOSE