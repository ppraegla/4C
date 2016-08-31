/*----------------------------------------------------------------------*/
/*!
 \file fluidporo_singlephaseDof.cpp

 \brief a material defining the degree of freedom of a single phase of
        a multiphase porous fluid

   \level 3

   \maintainer  Anh-Tu Vuong
                vuong@lnm.mw.tum.de
                http://www.lnm.mw.tum.de
                089 - 289-15251
 *----------------------------------------------------------------------*/

#include "fluidporo_singlephaseDof.H"

#include "fluidporo_singlephaselaw.H"

#include "matpar_bundle.H"

#include "../drt_lib/drt_globalproblem.H"

/*----------------------------------------------------------------------*
 *  constructor (public)                               vuong 08/16      |
 *----------------------------------------------------------------------*/
MAT::PAR::FluidPoroPhaseDof::FluidPoroPhaseDof(Teuchos::RCP<MAT::PAR::Material> matdata) :
  Parameter(matdata)
{
}

/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::PAR::FluidPoroPhaseDofDiffPressure::FluidPoroPhaseDofDiffPressure(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: FluidPoroPhaseDof(matdata),
  diffpresCoeffs_(matdata->Get<std::vector<int> >("PRESCOEFF")),
  phaselawId_(matdata->GetInt("PHASELAWID"))
{
  // retrieve problem instance to read from
  const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();

  // for the sake of safety
  if (DRT::Problem::Instance(probinst)->Materials() == Teuchos::null)
    dserror("Sorry dude, cannot work out problem instance.");
  // yet another safety check
  if (DRT::Problem::Instance(probinst)->Materials()->Num() == 0)
    dserror("Sorry dude, no materials defined.");

  // retrieve validated input line of material ID in question
  Teuchos::RCP<MAT::PAR::Material> curmat = DRT::Problem::Instance(probinst)->Materials()->ById(phaselawId_);

  // build the pressure-saturation law
  switch (curmat->Type())
  {
  case INPAR::MAT::m_fluidporo_phaselaw_linear:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::PAR::FluidPoroPhaseLawLinear(curmat));
    phaselaw_ = static_cast<MAT::PAR::FluidPoroPhaseLawLinear*>(curmat->Parameter());
    break;
  }
  case INPAR::MAT::m_fluidporo_phaselaw_tangent:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::PAR::FluidPoroPhaseLawTangent(curmat));
    phaselaw_ = static_cast<MAT::PAR::FluidPoroPhaseLawTangent*>(curmat->Parameter());
    break;
  }
  default:
    dserror("invalid pressure-saturation law for material %d", curmat->Type());
    break;
  }

  return;
}

/*----------------------------------------------------------------------*
 *  fill the dof matrix with the phase dofs                 vuong 08/16 |
*----------------------------------------------------------------------*/
void MAT::PAR::FluidPoroPhaseDofDiffPressure::FillDoFMatrix(
    Epetra_SerialDenseMatrix& dofmat,
    int numphase) const
{
  // safety check
  if((int)diffpresCoeffs_->size() != dofmat.N())
    dserror("Number of phases given by the poro singlephase material %i "
        "does not match number of DOFs (%i phases and %i DOFs)!",
        phaselaw_->Id(), diffpresCoeffs_->size(), dofmat.N());

  // fill pressure coefficients into matrix
  for(size_t i=0; i<diffpresCoeffs_->size();i++)
  {
    const int val = (*diffpresCoeffs_)[i];
    if(val!=0)
      dofmat(numphase,i) = val;
  }
}

/*----------------------------------------------------------------------*
 *  Evaluate generalized pressure of a phase                vuong 08/16 |
*----------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofDiffPressure::EvaluateGenPressure(
    int phasenum,
    const std::vector<double>& state) const
{
  // return the corresponding dof value
  return state[phasenum];
}


/*----------------------------------------------------------------------*
 *   Evaluate saturation of the phase                       vuong 08/16 |
*----------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofDiffPressure::EvaluateSaturation(
    int phasenum,
    const std::vector<double>& state,
    const std::vector<double>& pressure) const
{
  // call the phase law
  return  phaselaw_->EvaluateSaturation(pressure);
}


/*--------------------------------------------------------------------------*
 *  Evaluate derivative of saturation w.r.t. pressure           vuong 08/16 |
*---------------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofDiffPressure::EvaluateDerivOfSaturationWrtPressure(
    int phasenum,
    int doftoderive,
    const std::vector<double>& state) const
{
  // call the phase law
  return phaselaw_->EvaluateDerivOfSaturationWrtPressure(doftoderive,state);
}


/*----------------------------------------------------------------------------------------*
 * Evaluate derivative of degree of freedom with respect to pressure          vuong 08/16 |
*----------------------------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofDiffPressure::EvaluateDerivOfDofWrtPressure(
    int phasenum,
    int doftoderive,
    const std::vector<double>& state) const
{
  // derivative is the corresponding coefficient
  return  (*diffpresCoeffs_)[doftoderive];
}

/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::PAR::FluidPoroPhaseDofPressure::FluidPoroPhaseDofPressure(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: FluidPoroPhaseDof(matdata),
  phaselawId_(matdata->GetInt("PHASELAWID"))
{

  // retrieve problem instance to read from
  const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();

  // for the sake of safety
  if (DRT::Problem::Instance(probinst)->Materials() == Teuchos::null)
    dserror("Sorry dude, cannot work out problem instance.");
  // yet another safety check
  if (DRT::Problem::Instance(probinst)->Materials()->Num() == 0)
    dserror("Sorry dude, no materials defined.");

  // retrieve validated input line of material ID in question
  Teuchos::RCP<MAT::PAR::Material> curmat = DRT::Problem::Instance(probinst)->Materials()->ById(phaselawId_);

  // build the pressure-saturation law
  switch (curmat->Type())
  {
  case INPAR::MAT::m_fluidporo_phaselaw_linear:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::PAR::FluidPoroPhaseLawLinear(curmat));
    phaselaw_ = static_cast<MAT::PAR::FluidPoroPhaseLawLinear*>(curmat->Parameter());
    break;
  }
  case INPAR::MAT::m_fluidporo_phaselaw_tangent:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::PAR::FluidPoroPhaseLawTangent(curmat));
    phaselaw_ = static_cast<MAT::PAR::FluidPoroPhaseLawTangent*>(curmat->Parameter());
    break;
  }
  default:
    dserror("invalid pressure-saturation law for material %d", curmat->Type());
    break;
  }
  return;
}


/*----------------------------------------------------------------------*
 *  fill the dof matrix with the phase dofs                 vuong 08/16 |
*----------------------------------------------------------------------*/
void MAT::PAR::FluidPoroPhaseDofPressure::FillDoFMatrix(
    Epetra_SerialDenseMatrix& dofmat,
    int numphase) const
{
  // just mark the corresponding entry in the matrix
  dofmat(numphase,numphase) = 1.0;
}

/*----------------------------------------------------------------------*
 *  Evaluate generalized pressure of a phase                vuong 08/16 |
*----------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofPressure::EvaluateGenPressure(
    int phasenum,
    const std::vector<double>& state) const
{
  // return the corresponding dof value
  return state[phasenum];
}


/*----------------------------------------------------------------------*
 *   Evaluate saturation of the phase                       vuong 08/16 |
*----------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofPressure::EvaluateSaturation(
    int phasenum,
    const std::vector<double>& state,
    const std::vector<double>& pressure) const
{
  // call the phase law
  return phaselaw_->EvaluateSaturation(pressure);
}


/*--------------------------------------------------------------------------*
 *  Evaluate derivative of saturation w.r.t. pressure           vuong 08/16 |
*---------------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofPressure::EvaluateDerivOfSaturationWrtPressure(
    int phasenum,
    int doftoderive,
    const std::vector<double>& state) const
{
  // call the phase law
  return phaselaw_->EvaluateDerivOfSaturationWrtPressure(doftoderive,state);
}


/*----------------------------------------------------------------------------------------*
 * Evaluate derivative of degree of freedom with respect to pressure          vuong 08/16 |
*----------------------------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofPressure::EvaluateDerivOfDofWrtPressure(
    int phasenum,
    int doftoderive,
    const std::vector<double>& state) const
{
  double presurederiv = 0.0;

  // respective derivative of w.r.t. is either 0 or 1
  if(phasenum==doftoderive)
    presurederiv = 1.0;

  return presurederiv;
}


/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::PAR::FluidPoroPhaseDofPressureSum::FluidPoroPhaseDofPressureSum(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: FluidPoroPhaseDof(matdata)
{
}

/*----------------------------------------------------------------------*
 *  fill the dof matrix with the phase dofs                 vuong 08/16 |
*----------------------------------------------------------------------*/
void MAT::PAR::FluidPoroPhaseDofPressureSum::FillDoFMatrix(
    Epetra_SerialDenseMatrix& dofmat,
    int numphase) const
{
  // just mark the corresponding entry in the matrix
  dofmat(numphase,numphase) = 1.0;
}

/*----------------------------------------------------------------------*
 *  Evaluate generalized pressure of a phase                vuong 08/16 |
*----------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofPressureSum::EvaluateGenPressure(
    int phasenum,
    const std::vector<double>& state) const
{
  // return the corresponding dof value
  return state[phasenum];
}


/*----------------------------------------------------------------------*
 *   Evaluate saturation of the phase                       vuong 08/16 |
*----------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofPressureSum::EvaluateSaturation(
    int phasenum,
    const std::vector<double>& state,
    const std::vector<double>& pressure) const
{
  // the saturation is calculated from the other phases -> the phase manager class handles this
  dserror("The saturation of the last phase needs to be computed form the sum of all other saturations");

  return 0.0;
}


/*--------------------------------------------------------------------------*
 *  Evaluate derivative of saturation w.r.t. pressure           vuong 08/16 |
*---------------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofPressureSum::EvaluateDerivOfSaturationWrtPressure(
    int phasenum,
    int doftoderive,
    const std::vector<double>& state) const
{
  // the derivative is determined by the other phases,
  // as the saturation is given as 1.0 - (sum of saturation of all other phases)
  // It is evaluated by the phase manager class
  dserror("The saturation of the last phase needs to be computed form the sum of all other saturations");

  return 0.0;
}


/*----------------------------------------------------------------------------------------*
 * Evaluate derivative of degree of freedom with respect to pressure          vuong 08/16 |
*----------------------------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofPressureSum::EvaluateDerivOfDofWrtPressure(
    int phasenum,
    int doftoderive,
    const std::vector<double>& state) const
{
  double presurederiv = 0.0;

  // respective derivative of w.r.t. is either 0 or 1
  if(phasenum==doftoderive)
    presurederiv = 1.0;

  return presurederiv;
}

/************************************************************************/
/************************************************************************/

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
MAT::PAR::FluidPoroPhaseDofSaturation::FluidPoroPhaseDofSaturation(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: FluidPoroPhaseDof(matdata),
  phaselawId_(matdata->GetInt("PHASELAWID"))
{

  // retrieve problem instance to read from
  const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();

  // for the sake of safety
  if (DRT::Problem::Instance(probinst)->Materials() == Teuchos::null)
    dserror("Sorry dude, cannot work out problem instance.");
  // yet another safety check
  if (DRT::Problem::Instance(probinst)->Materials()->Num() == 0)
    dserror("Sorry dude, no materials defined.");

  // retrieve validated input line of material ID in question
  Teuchos::RCP<MAT::PAR::Material> curmat = DRT::Problem::Instance(probinst)->Materials()->ById(phaselawId_);

  // build the pressure-saturation law
  switch (curmat->Type())
  {
  case INPAR::MAT::m_fluidporo_phaselaw_linear:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::PAR::FluidPoroPhaseLawLinear(curmat));
    phaselaw_ = static_cast<MAT::PAR::FluidPoroPhaseLawLinear*>(curmat->Parameter());
    break;
  }
  case INPAR::MAT::m_fluidporo_phaselaw_tangent:
  {
    if (curmat->Parameter() == NULL)
      curmat->SetParameter(new MAT::PAR::FluidPoroPhaseLawTangent(curmat));
    phaselaw_ = static_cast<MAT::PAR::FluidPoroPhaseLawTangent*>(curmat->Parameter());
    break;
  }
  default:
    dserror("invalid pressure-saturation law for material %d", curmat->Type());
    break;
  }
  return;
}

/*----------------------------------------------------------------------*
 *  fill the dof matrix with the phase dofs                 vuong 08/16 |
*----------------------------------------------------------------------*/
void MAT::PAR::FluidPoroPhaseDofSaturation::FillDoFMatrix(
    Epetra_SerialDenseMatrix& dofmat,
    int numphase) const
{
  // get pressure coefficients of phase law
  const std::vector<int>* presIDs = phaselaw_->presids_;

  // safety check
  if((int)presIDs->size() != dofmat.N())
    dserror("Number of phases given by the poro phase law material %i "
        "does not match number of DOFs (%i phases and %i DOFs)!",
        phaselaw_->Id(), presIDs->size(), dofmat.N());

  // fill pressure coefficients of phase law into matrix
  for(size_t i=0; i<presIDs->size();i++)
  {
    const int val = (*presIDs)[i];
    if(val!=0)
      dofmat(numphase,i) = val;
  }
}

/*----------------------------------------------------------------------*
 *  Evaluate generalized pressure of a phase                vuong 08/16 |
*----------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofSaturation::EvaluateGenPressure(
    int phasenum,
    const std::vector<double>& state) const
{
  // evaluate the phase law for the generalized (i.e. some differential pressure)
  // the phase law depends on
  return phaselaw_->EvaluateGenPressure(state[phasenum]);
}


/*----------------------------------------------------------------------*
 *   Evaluate saturation of the phase                       vuong 08/16 |
*----------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofSaturation::EvaluateSaturation(
    int phasenum,
    const std::vector<double>& state,
    const std::vector<double>& pressure) const
{
  // get the corresponding dof value
  return state[phasenum];
}


/*--------------------------------------------------------------------------*
 *  Evaluate derivative of saturation w.r.t. pressure           vuong 08/16 |
*---------------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofSaturation::EvaluateDerivOfSaturationWrtPressure(
    int phasenum,
    int doftoderive,
    const std::vector<double>& state) const
{
  // call the phase law
  return phaselaw_->EvaluateDerivOfSaturationWrtPressure(doftoderive,state);
}


/*----------------------------------------------------------------------------------------*
 * Evaluate derivative of degree of freedom with respect to pressure          vuong 08/16 |
*----------------------------------------------------------------------------------------*/
double MAT::PAR::FluidPoroPhaseDofSaturation::EvaluateDerivOfDofWrtPressure(
    int phasenum,
    int doftoderive,
    const std::vector<double>& state) const
{
  // call the phase law for the derivative
  return phaselaw_->EvaluateDerivOfSaturationWrtPressure(doftoderive,state);
}

/************************************************************************/
/************************************************************************/