/*-----------------------------------------------------------*/
/*!
\file str_impl_statics.cpp

\maintainer Michael Hiermeier

\date Nov 30, 2015

\level 3

*/
/*-----------------------------------------------------------*/

#include "str_impl_statics.H"
#include "str_model_evaluator.H"
#include "str_timint_implicit.H"
#include "str_predict_generic.H"
#include "str_dbc.H"

#include "../linalg/linalg_sparseoperator.H"

#include <Epetra_Vector.h>
#include <NOX_Epetra_Vector.H>

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::IMPLICIT::Statics::Statics()
{
  // empty constructor
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::Setup()
{
  CheckInit();

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::SetState(const Epetra_Vector& x)
{
  CheckInitSetup();
  GlobalState().GetMutableDisNp() = GlobalState().ExportDisplEntries(x);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::ApplyForce(const Epetra_Vector& x,
    Epetra_Vector& f)
{
  CheckInitSetup();
  return ModelEval().ApplyForce(x,f);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::ApplyStiff(const Epetra_Vector& x,
    LINALG::SparseOperator& jac)
{
  CheckInitSetup();
  return ModelEval().ApplyStiff(x,jac);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::ApplyForceStiff(const Epetra_Vector& x,
    Epetra_Vector& f, LINALG::SparseOperator& jac)
{
  CheckInitSetup();
  return ModelEval().ApplyForceStiff(x,f,jac);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::IMPLICIT::Statics::CalcRefNormForce(
    const enum NOX::Abstract::Vector::NormType& type)
{
  CheckInitSetup();
  // switch from Epetra_Vector to NOX::Epetra::Vector (view but read-only)
  Teuchos::RCP<const NOX::Epetra::Vector> fintnp_nox_ptr =
      Teuchos::rcp(new NOX::Epetra::Vector(GlobalState().GetMutableFintNp(),
          NOX::Epetra::Vector::CreateView));
  Teuchos::RCP<const NOX::Epetra::Vector> fextnp_nox_ptr =
        Teuchos::rcp(new NOX::Epetra::Vector(GlobalState().GetMutableFextNp(),
            NOX::Epetra::Vector::CreateView));
  Teuchos::RCP<const NOX::Epetra::Vector> freactnp_nox_ptr =
        Teuchos::rcp(new NOX::Epetra::Vector(GlobalState().GetMutableFreactNp(),
            NOX::Epetra::Vector::CreateView));

  // norm of the internal forces
  double fintnorm = fintnp_nox_ptr->norm(type);

  // norm of the external forces
  double fextnorm = fextnp_nox_ptr->norm(type);

  // norm of reaction forces
  double freactnorm = freactnp_nox_ptr->norm(type);

  // return characteristic norm
  return std::max(fintnorm, std::max(fextnorm, freactnorm));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::IMPLICIT::Statics::GetIntParam() const
{
  CheckInitSetup();
  return 0.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::PreUpdate()
{
  CheckInitSetup();
  const STR::TIMINT::Implicit* impl_ptr =
      dynamic_cast<const STR::TIMINT::Implicit*>(&TimInt());
  if (impl_ptr == NULL)
    return;

  // get the time step size
  const double dt = (*GlobalState().GetDeltaTime())[0];

  const INPAR::STR::PredEnum& pred_type = impl_ptr->Predictor().GetType();
  Teuchos::RCP<Epetra_Vector>& accnp_ptr = GlobalState().GetMutableAccNp();
  Teuchos::RCP<Epetra_Vector>& velnp_ptr = GlobalState().GetMutableVelNp();

  switch (pred_type)
  {
    // case: constant acceleration
    case INPAR::STR::pred_constacc:
    {
      // read-only access
      Teuchos::RCP<const Epetra_Vector> veln_ptr = GlobalState().GetVelN();
      // update the pseudo acceleration
      accnp_ptr->Update(1.0/dt,*velnp_ptr,-1.0/dt,*veln_ptr,0.0);
    }
    // case: constant acceleration OR constant velocity
    case INPAR::STR::pred_constvel:
    {
      // read-only access
      Teuchos::RCP<const Epetra_Vector> disn_ptr  = GlobalState().GetDisN();
      Teuchos::RCP<const Epetra_Vector> disnp_ptr = GlobalState().GetDisNp();
      // update the pseudo velocity (statics!)
      velnp_ptr->Update(1.0/dt,*disnp_ptr,-1.0/dt,*disn_ptr,0.0);
      // ATTENTION: Break for both cases!
      break;
    }
    default:
      /* do nothing */
      break;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::UpdateStepState()
{
  CheckInitSetup();
  // update model specific variables
  ModelEval().UpdateStepState();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::UpdateStepElement()
{
  CheckInitSetup();
  ModelEval().UpdateStepElement();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::OutputStepState()
{
  CheckInitSetup();
  ModelEval().OutputStepState();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::IMPLICIT::Statics::PredictConstDisConsistVelAcc(
    Epetra_Vector& disnp,
    Epetra_Vector& velnp,
    Epetra_Vector& accnp) const
{
  CheckInitSetup();
  // constant predictor : displacement in domain
  disnp.Update(1.0,*GlobalState().GetDisN(),0.0);
  // new end-point velocities, these stay zero in static calculation
  velnp.PutScalar(0.0);
  // new end-point accelerations, these stay zero in static calculation
  accnp.PutScalar(0.0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::PredictConstVelConsistAcc(
    Epetra_Vector& disnp,
    Epetra_Vector& velnp,
    Epetra_Vector& accnp) const
{
  CheckInitSetup();
  // If there is not enough history information, return a fail status.
  if (GlobalState().GetStepN()==0)
    return false;

  // Displacement increment over last time step
  Teuchos::RCP<Epetra_Vector> disp_inc =
      Teuchos::rcp(new Epetra_Vector(*GlobalState().DofRowMapView(), true));
  disp_inc->Update((*GlobalState().GetDeltaTime())[0],*GlobalState().GetVelN(),0.);
  // apply the dbc on the auxiliary vector
  TimInt().GetDBC().ApplyDirichletToVector(disp_inc);
  // update the solution variables
  disnp.Update(1.0,*GlobalState().GetDisN(),0.0);
  disnp.Update(1.0,*disp_inc,1.0);
  velnp.Update(1.0,*GlobalState().GetVelN(),0.0);
  accnp.Update(1.0,*GlobalState().GetAccN(),0.0);

  return true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool STR::IMPLICIT::Statics::PredictConstAcc(
    Epetra_Vector& disnp,
    Epetra_Vector& velnp,
    Epetra_Vector& accnp) const
{
  CheckInitSetup();
  // If there is not enough history information try a different predictor with
  // less requirements.
  if (GlobalState().GetStepN()<2)
    return PredictConstVelConsistAcc(disnp,velnp,accnp);

  // Displacement increment over last time step
  Teuchos::RCP<Epetra_Vector> disp_inc =
      Teuchos::rcp(new Epetra_Vector(*GlobalState().DofRowMapView(), true));
  const double& dt = (*GlobalState().GetDeltaTime())[0];
  disp_inc->Update(dt,*GlobalState().GetVelN(),0.);
  disp_inc->Update(0.5*dt*dt,*GlobalState().GetAccN(),1.0);
  // apply the dbc on the auxiliary vector
  TimInt().GetDBC().ApplyDirichletToVector(disp_inc);
  // update the solution variables
  disnp.Update(1.0, *GlobalState().GetDisN(), 0.0);
  disnp.Update(1.,*disp_inc,1.);
  velnp.Update(1.0, *GlobalState().GetVelN(), 0.0);
  accnp.Update(1.0, *GlobalState().GetAccN(), 0.0);

  return true;
}