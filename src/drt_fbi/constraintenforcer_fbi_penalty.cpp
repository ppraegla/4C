/*----------------------------------------------------------------------*/
/*! \file

\brief Implements the constraint enforcement technique of a penalty approach (Mortar and GPTS) (for
fluid-beam interaction)

\level 2

\maintainer Nora Hagmeyer
*----------------------------------------------------------------------*/

#include "ad_fbi_constraintbridge_penalty.H"
#include "beam_to_fluid_meshtying_params.H"
#include "constraintenforcer_fbi_penalty.H"
#include "constraintenforcer_fbi.H"
#include "../drt_adapter/ad_str_fbiwrapper.H"
#include "../drt_adapter/ad_fld_moving_boundary.H"
#include "../linalg/linalg_sparsematrix.H"
#include <Epetra_Vector.h>

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

Teuchos::RCP<LINALG::SparseMatrix>
ADAPTER::FBIPenaltyConstraintenforcer::AssembleFluidCouplingMatrix() const
{
  // Get coupling contributions to the fluid stiffness matrix and scale them with the penalty
  // parameter
  if (Teuchos::rcp_dynamic_cast<ADAPTER::FBIConstraintBridgePenalty>(GetBridge(), true)
          ->GetCff()
          ->Scale(GetBridge()->GetParams()->GetPenaltyParameter()))
    dserror("Scaling of the penalty stiffness was unsuccessful!\n");

  return Teuchos::rcp_dynamic_cast<ADAPTER::FBIConstraintBridgePenalty>(GetBridge(), true)
      ->GetCff();
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<LINALG::SparseMatrix>
ADAPTER::FBIPenaltyConstraintenforcer::AssembleStructureCouplingMatrix() const
{
  // For the classical partitioned algorithm we do not have any contributions to the stiffness
  // matrix of the structure field
  return Teuchos::null;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> ADAPTER::FBIPenaltyConstraintenforcer::AssembleFluidCouplingResidual()
    const
{
  // Get the force acting on the fluid field, scale it with the penalty factor and -1 to get the
  // correct direction
  Teuchos::RCP<Epetra_Vector> f = Teuchos::rcp(new Epetra_Vector(
      (Teuchos::rcp_dynamic_cast<ADAPTER::FBIConstraintBridgePenalty>(GetBridge(), true)
              ->GetFluidCouplingResidual())
          ->Map()));
  f->Update(-1.0,
      *(Teuchos::rcp_dynamic_cast<ADAPTER::FBIConstraintBridgePenalty>(GetBridge(), true)
              ->GetFluidCouplingResidual()),
      0.0);
  if (f->Scale(GetBridge()->GetParams()->GetPenaltyParameter()))
    dserror("Scaling of the penalty force was unsuccessful!\n");
  return f;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector>
ADAPTER::FBIPenaltyConstraintenforcer::AssembleStructureCouplingResidual() const
{
  // Get the force acting on the structure field, scale it with the penalty factor and -1 to get the
  // correct direction
  Teuchos::RCP<Epetra_Vector> f = Teuchos::rcp(new Epetra_Vector(
      (Teuchos::rcp_dynamic_cast<ADAPTER::FBIConstraintBridgePenalty>(GetBridge(), true)
              ->GetStructureCouplingResidual())
          ->Map()));
  f->Update(-1.0,
      *(Teuchos::rcp_dynamic_cast<ADAPTER::FBIConstraintBridgePenalty>(GetBridge(), true)
              ->GetStructureCouplingResidual()),
      0.0);
  if (f->Scale(GetBridge()->GetParams()->GetPenaltyParameter()))
    dserror("Scaling of the penalty force was unsuccessful!\n");

  return f;
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ADAPTER::FBIPenaltyConstraintenforcer::PrepareFluidSolve()
{
  GetBridge()->PrepareFluidSolve();
}