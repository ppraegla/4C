/*----------------------------------------------------------------------*/
/*! \file
\brief multiscale functionality for pyramid shaped solid element
\level 2


*----------------------------------------------------------------------*/


#include "baci_comm_utils.H"
#include "baci_lib_discret.H"
#include "baci_lib_globalproblem.H"
#include "baci_mat_micromaterial.H"
#include "baci_so3_pyramid5.H"



/*----------------------------------------------------------------------*
 |  homogenize material density (public)                                |
 *----------------------------------------------------------------------*/
// this routine is intended to determine a homogenized material
// density for multi-scale analyses by averaging over the initial volume

void DRT::ELEMENTS::So_pyramid5::sop5_homog(Teuchos::ParameterList& params)
{
  if (DRT::Problem::Instance(0)->GetCommunicators()->SubComm()->MyPID() == Owner())
  {
    double homogdens = 0.;
    const static std::vector<double> weights = sop5_weights();

    for (int gp = 0; gp < NUMGPT_SOP5; ++gp)
    {
      const double density = Material()->Density(gp);
      homogdens += detJ_[gp] * weights[gp] * density;
    }

    double homogdensity = params.get<double>("homogdens", 0.0);
    params.set("homogdens", homogdensity + homogdens);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Read restart on the microscale                                      |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_pyramid5::sop5_read_restart_multi()
{
  Teuchos::RCP<MAT::Material> mat = Material();

  if (mat->MaterialType() == INPAR::MAT::m_struct_multiscale)
  {
    auto* micro = dynamic_cast<MAT::MicroMaterial*>(mat.get());
    int eleID = Id();
    bool eleowner = false;
    if (DRT::Problem::Instance()->GetDis("structure")->Comm().MyPID() == Owner()) eleowner = true;

    for (int gp = 0; gp < NUMGPT_SOP5; ++gp) micro->ReadRestart(gp, eleID, eleowner);
  }

  return;
}