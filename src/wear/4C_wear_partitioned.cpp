/*----------------------------------------------------------------------*/
/*! \file

\brief  Basis of all structure approaches with ale
        (Lagrangian step followed by Shape Evolution step )

\level 2


*/
/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | headers                                                  farah 11/13 |
 *----------------------------------------------------------------------*/
#include "4C_wear_partitioned.hpp"

#include "4C_adapter_ale_wear.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_contact_abstract_strategy.hpp"
#include "4C_contact_defines.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_friction_node.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_interface.hpp"
#include "4C_contact_lagrange_strategy_wear.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_wear_interface.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_fs3i_biofilm_fsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_ale.hpp"
#include "4C_inpar_wear.hpp"
#include "4C_lib_element.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mortar_manager_base.hpp"
#include "4C_so3_hex20.hpp"
#include "4C_so3_hex27.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_tet10.hpp"
#include "4C_so3_tet4.hpp"
#include "4C_structure_aux.hpp"
#include "4C_utils_parameter_list.hpp"
#include "4C_w1.hpp"
#include "4C_wear_utils.hpp"

#include <Epetra_SerialComm.h>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | constructor (public)                                     farah 05/13 |
 *----------------------------------------------------------------------*/
WEAR::Partitioned::Partitioned(const Epetra_Comm& comm) : Algorithm(comm)
{
  const int ndim = GLOBAL::Problem::Instance()->NDim();

  // create ale-struct coupling
  const Epetra_Map* structdofmap = StructureField()->Discretization()->NodeRowMap();
  const Epetra_Map* aledofmap = AleField().Discretization()->NodeRowMap();

  if (CORE::UTILS::IntegralValue<bool>(GLOBAL::Problem::Instance()->WearParams(), "MATCHINGGRID"))
  {
    // if there are two identical nodes (i.e. for initial contact) the nodes matching creates an
    // error !!!
    coupalestru_ = Teuchos::rcp(new CORE::ADAPTER::Coupling());
    Teuchos::rcp_dynamic_cast<CORE::ADAPTER::Coupling>(coupalestru_)
        ->SetupCoupling(*AleField().Discretization(), *StructureField()->Discretization(),
            *aledofmap, *structdofmap, ndim);
  }
  else
  {
    // Scheme: non matching meshes --> volumetric mortar coupling...
    coupalestru_ = Teuchos::rcp(new CORE::ADAPTER::MortarVolCoupl());

    // projection ale -> structure : all ndim dofs (displacements)
    std::vector<int> coupleddof12 = std::vector<int>(ndim, 1);

    // projection structure -> ale : all ndim dofs (displacements)
    std::vector<int> coupleddof21 = std::vector<int>(ndim, 1);

    std::pair<int, int> dofset12(0, 0);
    std::pair<int, int> dofset21(0, 0);

    // init coupling
    Teuchos::rcp_dynamic_cast<CORE::ADAPTER::MortarVolCoupl>(coupalestru_)
        ->Init(ndim, GLOBAL::Problem::Instance()->GetDis("ale"),
            GLOBAL::Problem::Instance()->GetDis("structure"), &coupleddof12, &coupleddof21,
            &dofset12, &dofset21, Teuchos::null, false);

    // redistribute discretizations to meet needs of volmortar coupling
    //    Teuchos::rcp_dynamic_cast<ADAPTER::MortarVolCoupl>(coupalestru_)->Redistribute();

    // setup projection matrices
    Teuchos::rcp_dynamic_cast<CORE::ADAPTER::MortarVolCoupl>(coupalestru_)
        ->Setup(GLOBAL::Problem::Instance()->VolmortarParams());
  }

  // create interface coupling
  coupstrualei_ = Teuchos::rcp(new CORE::ADAPTER::Coupling());
  coupstrualei_->SetupConditionCoupling(*StructureField()->Discretization(),
      StructureField()->Interface()->AleWearCondMap(), *AleField().Discretization(),
      AleField().Interface()->Map(AleField().Interface()->cond_ale_wear), "AleWear", ndim);

  // initialize intern variables for wear
  wearnp_i_ = Teuchos::rcp(
      new Epetra_Vector(*AleField().Interface()->Map(AleField().Interface()->cond_ale_wear)), true);
  wearnp_ip_ = Teuchos::rcp(
      new Epetra_Vector(*AleField().Interface()->Map(AleField().Interface()->cond_ale_wear)), true);
  wearincr_ = Teuchos::rcp(
      new Epetra_Vector(*AleField().Interface()->Map(AleField().Interface()->cond_ale_wear)), true);
  delta_ale_ = Teuchos::rcp(new Epetra_Vector(AleField().Dispnp()->Map(), true));
  ale_i_ = Teuchos::rcp(new Epetra_Vector(AleField().Dispnp()->Map(), true));

  alepara_ = GLOBAL::Problem::Instance()->AleDynamicParams();
}


/*----------------------------------------------------------------------*
 | general time loop                                        farah 10/13 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::TimeLoop()
{
  // get wear paramter list
  const Teuchos::ParameterList& wearpara = GLOBAL::Problem::Instance()->WearParams();
  double timeratio = wearpara.get<double>("WEAR_TIMERATIO");

  int counter = -1;
  bool alestep = false;

  // time loop
  while (NotFinished())
  {
    if ((int)(Step() / timeratio) > counter)
    {
      counter++;
      alestep = true;
    }

    if (CORE::UTILS::IntegralValue<INPAR::WEAR::WearCoupAlgo>(wearpara, "WEAR_COUPALGO") ==
        INPAR::WEAR::wear_stagg)
      TimeLoopStagg(alestep);
    else if (CORE::UTILS::IntegralValue<INPAR::WEAR::WearCoupAlgo>(wearpara, "WEAR_COUPALGO") ==
             INPAR::WEAR::wear_iterstagg)
      TimeLoopIterStagg();
    else
      FOUR_C_THROW("WEAR::TimeLoop: Algorithm not provided!");

    alestep = false;
  }  // time loop
}


/*----------------------------------------------------------------------*
 | time loop for staggered coupling                         farah 11/13 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::TimeLoopIterStagg()
{
  // counter and print header
  IncrementTimeAndStep();
  PrintHeader();

  // prepare time step for both fields
  PrepareTimeStep();

  bool converged = false;  // converged state?
  int iter = 0;            // iteration counter

  // stactic cast of mortar strategy to contact strategy
  MORTAR::StrategyBase& strategy = cmtman_->GetStrategy();
  WEAR::LagrangeStrategyWear& cstrategy = static_cast<WEAR::LagrangeStrategyWear&>(strategy);

  // reset waccu, wold and wcurr...
  cstrategy.UpdateWearDiscretIterate(false);

  /*************************************************************
   * Nonlinear iterations between Structure and ALE:           *
   * 1. Solve structure + contact to get wear                  *
   * 2. Apply wear increment (i+1 - i) onto ALE (add function) *
   * 3. Employ ALE disp incr (i+1 - i) and spat disp i to get  *
   *    abs mat disp for timestep n+1                          *
   * 4. Upadate spat disp from i to i+1                        *
   * 5. Check for convergence                                  *
   * 6. store ALE disp i = i+1                                 *
   *************************************************************/
  while (converged == false)
  {
    // 1. solution
    StructureField()->Solve();

    // 2. wear as interface displacements in ale dofs
    Teuchos::RCP<Epetra_Vector> idisale_s, idisale_m;
    InterfaceDisp(idisale_s, idisale_m);

    // merge the both wear vectors for master and slave side to one global vector
    MergeWear(idisale_s, idisale_m, wearincr_);

    // coupling of struct/mortar and ale dofs
    DispCoupling(wearincr_);

    if (Comm().MyPID() == 0)
      std::cout << "========================= ALE STEP =========================" << std::endl;

    // do ale step
    AleStep(wearincr_);

    // 3. application of mesh displacements to structural field,
    // update material displacements
    UpdateMatConf();

    // 4. update dispnp
    UpdateSpatConf();

    // 5. convergence check fot current iteration
    converged = ConvergenceCheck(iter);

    // store old wear
    cstrategy.UpdateWearDiscretIterate(true);

    ++iter;
  }  // end nonlin loop

  // update for structure and ale
  Update();

  // output for structure and ale
  Output();

  return;
}


/*----------------------------------------------------------------------*
 | time loop for oneway coupling                            farah 11/13 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::TimeLoopStagg(bool alestep)
{
  // stactic cast of mortar strategy to contact strategy
  MORTAR::StrategyBase& strategy = cmtman_->GetStrategy();
  WEAR::LagrangeStrategyWear& cstrategy = static_cast<WEAR::LagrangeStrategyWear&>(strategy);

  // counter and print header
  IncrementTimeAndStep();
  PrintHeader();

  // prepare time step for both fields
  PrepareTimeStep();

  /********************************************************************/
  /* START LAGRANGE STEP                                              */
  /* structural lagrange step with contact                            */
  /********************************************************************/

  // solution
  StructureField()->Solve();

  if (alestep)
  {
    if (Comm().MyPID() == 0)
      std::cout << "========================= ALE STEP =========================" << std::endl;

    /********************************************************************/
    /* COUPLING                                                         */
    /* Wear from structure solve as dirichlet for ALE                   */
    /********************************************************************/

    // wear as interface displacements in interface dofs
    Teuchos::RCP<Epetra_Vector> idisale_s, idisale_m, idisale_global;
    InterfaceDisp(idisale_s, idisale_m);

    // merge the both wear vectors for master and slave side to one global vector
    MergeWear(idisale_s, idisale_m, idisale_global);

    // coupling of struct/mortar and ale dofs
    DispCoupling(idisale_global);

    /********************************************************************/
    /* Shape Evolution STEP                                             */
    /* 1. mesh displacements due to wear from ALE system                */
    /* 2. mapping of results from "old" to "new" mesh                   */
    /********************************************************************/

    // do ale step
    AleStep(idisale_global);

    // update material displacements
    UpdateMatConf();

    // update spatial displacements
    UpdateSpatConf();

    // reset wear
    cstrategy.UpdateWearDiscretIterate(false);
  }
  else
  {
    cstrategy.UpdateWearDiscretAccumulation();
  }

  /********************************************************************/
  /* FINISH STEP:                                                     */
  /* Update and Write Output                                          */
  /********************************************************************/

  // update for structure and ale
  Update();

  // output for structure and ale
  Output();

  return;
}


/*----------------------------------------------------------------------*
 | prepare time step for ale and structure                  farah 11/13 |
 *----------------------------------------------------------------------*/
bool WEAR::Partitioned::ConvergenceCheck(int iter)
{
  double Wincr = 0.0;
  double ALEincr = 0.0;
  wearincr_->Norm2(&Wincr);
  delta_ale_->Norm2(&ALEincr);

  if (Comm().MyPID() == 0)
  {
    std::cout << "-----------------"
              << " Step " << iter + 1 << " --------------------" << std::endl;
    std::cout << "Wear incr.= " << Wincr << "         ALE incr.= " << ALEincr << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
  }

  // TODO tolerance from input!!!
  // check loads
  if (abs(Wincr) < 1e-8 and abs(ALEincr) < 1e-8)
  {
    // reset vectors
    ale_i_->PutScalar(0.0);
    delta_ale_->PutScalar(0.0);

    return true;
  }

  if (iter > 50)
    FOUR_C_THROW(
        "Staggered solution scheme for ale-wear problem unconverged within 50 nonlinear iteration "
        "steps!");

  return false;
}


/*----------------------------------------------------------------------*
 | prepare time step for ale and structure                  farah 11/13 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::PrepareTimeStep()
{
  // predict and solve structural system
  StructureField()->PrepareTimeStep();

  // prepare ale output: increase time step
  AleField().PrepareTimeStep();

  return;
}


/*----------------------------------------------------------------------*
 | update ale and structure                                 farah 11/13 |
 *---------------------------------------------------------------- ------*/
void WEAR::Partitioned::Update()
{
  // update at time step
  StructureField()->Update();

  // update
  AleField().Update();

  return;
}


/*----------------------------------------------------------------------*
 | update spatial displacements                             farah 05/14 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::UpdateSpatConf()
{
  // mesh displacement from solution of ALE field in structural dofs
  // first perform transformation from ale to structure dofs
  Teuchos::RCP<Epetra_Vector> disalenp = AleToStructure(AleField().Dispnp());
  Teuchos::RCP<Epetra_Vector> disalen = AleToStructure(AleField().Dispn());

  // get structure dispnp vector
  Teuchos::RCP<Epetra_Vector> dispnp =
      StructureField()->WriteAccessDispnp();  // change to ExtractDispn() for overlap

  // get info about wear conf
  INPAR::WEAR::WearShapeEvo wconf = CORE::UTILS::IntegralValue<INPAR::WEAR::WearShapeEvo>(
      GLOBAL::Problem::Instance()->WearParams(), "WEAR_SHAPE_EVO");

  // for shape evol in spat conf
  if (wconf == INPAR::WEAR::wear_se_sp)
  {
    int err = 0;
    // update per absolute vector
    err = dispnp->Update(1.0, *disalenp, 0.0);
    if (err != 0) FOUR_C_THROW("update wrong!");
  }
  // for shape evol in mat conf
  else if (wconf == INPAR::WEAR::wear_se_mat)
  {
    // set state
    (StructureField()->Discretization())->SetState(0, "displacement", dispnp);

    // set state
    (StructureField()->Discretization())
        ->SetState(0, "material_displacement", StructureField()->DispMat());

    // loop over all row nodes to fill graph
    for (int k = 0; k < StructureField()->Discretization()->NumMyRowNodes(); ++k)
    {
      int gid = StructureField()->Discretization()->NodeRowMap()->GID(k);
      DRT::Node* node = StructureField()->Discretization()->gNode(gid);
      DRT::Element** ElementPtr = node->Elements();
      int numelement = node->NumElement();

      const int numdof = StructureField()->Discretization()->NumDof(node);

      // create Xmat for 3D problems
      std::vector<double> Xspatial(numdof);
      std::vector<double> Xmat(numdof);

      for (int dof = 0; dof < numdof; ++dof)
      {
        int dofgid = StructureField()->Discretization()->Dof(node, dof);
        int doflid = (dispnp->Map()).LID(dofgid);
        Xmat[dof] = node->X()[dof] + (*StructureField()->DispMat())[doflid];
      }

      // create updated  Xspatial --> via nonlinear interpolation between nodes (like gp projection)
      AdvectionMap(Xspatial.data(), Xmat.data(), ElementPtr, numelement, false);

      // store in dispmat
      for (int dof = 0; dof < numdof; ++dof)
      {
        int dofgid = StructureField()->Discretization()->Dof(node, dof);
        int doflid = (dispnp->Map()).LID(dofgid);
        (*dispnp)[doflid] = Xspatial[dof] - node->X()[dof];
      }
    }  // end row node loop
  }
  else
  {
    FOUR_C_THROW("Unknown wear configuration!");
  }

  return;
}


/*----------------------------------------------------------------------*
 | output ale and structure                                 farah 11/13 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::Output()
{
  // calculate stresses, strains, energies
  constexpr bool force_prepare = false;
  StructureField()->PrepareOutput(force_prepare);

  // write strcture output to screen and files
  StructureField()->Output();

  // output ale
  AleField().Output();

  return;
}


/*----------------------------------------------------------------------*
 | Perform Coupling from struct/mortar to ale dofs          farah 05/13 |
 | This is necessary due to the parallel redistribution                 |
 | of the contact interface                                             |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::DispCoupling(Teuchos::RCP<Epetra_Vector>& disinterface)
{
  // Teuchos::RCP<Epetra_Vector> aledofs = Teuchos::rcp(new
  // Epetra_Vector(*AleField().Interface()->Map(AleField().Interface()->cond_ale_wear)),true);
  Teuchos::RCP<Epetra_Vector> strudofs =
      Teuchos::rcp(new Epetra_Vector(*StructureField()->Interface()->AleWearCondMap()), true);

  // change the parallel distribution from mortar interface to structure
  CORE::LINALG::Export(*disinterface, *strudofs);

  // perform coupling to ale dofs
  disinterface.reset();
  disinterface = coupstrualei_->MasterToSlave(strudofs);

  return;
}


/*----------------------------------------------------------------------*
 | Merge wear from slave and master surface to one          farah 06/13 |
 | wear vector                                                          |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::MergeWear(Teuchos::RCP<Epetra_Vector>& disinterface_s,
    Teuchos::RCP<Epetra_Vector>& disinterface_m, Teuchos::RCP<Epetra_Vector>& disinterface_g)
{
  MORTAR::StrategyBase& strategy = cmtman_->GetStrategy();
  CONTACT::AbstractStrategy& cstrategy = static_cast<CONTACT::AbstractStrategy&>(strategy);
  std::vector<Teuchos::RCP<CONTACT::Interface>> interface = cstrategy.ContactInterfaces();
  Teuchos::RCP<WEAR::WearInterface> winterface =
      Teuchos::rcp_dynamic_cast<WEAR::WearInterface>(interface[0]);
  if (winterface == Teuchos::null) FOUR_C_THROW("Casting to WearInterface returned null!");

  disinterface_g = Teuchos::rcp(new Epetra_Vector(*winterface->Discret().DofRowMap()), true);
  Teuchos::RCP<Epetra_Vector> auxvector =
      Teuchos::rcp(new Epetra_Vector(*winterface->Discret().DofRowMap()), true);

  CORE::LINALG::Export(*disinterface_s, *disinterface_g);
  CORE::LINALG::Export(*disinterface_m, *auxvector);

  int err = 0;
  err = disinterface_g->Update(1.0, *auxvector, true);
  if (err != 0) FOUR_C_THROW("update wrong!");

  return;
}


/*----------------------------------------------------------------------*
 | Vector of interface displacements in ALE dofs            farah 05/13 |
 | Currently just for 1 interface                                       |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::InterfaceDisp(
    Teuchos::RCP<Epetra_Vector>& disinterface_s, Teuchos::RCP<Epetra_Vector>& disinterface_m)
{
  // get info about wear side
  INPAR::WEAR::WearSide wside = CORE::UTILS::IntegralValue<INPAR::WEAR::WearSide>(
      GLOBAL::Problem::Instance()->WearParams(), "WEAR_SIDE");

  // get info about wear type
  INPAR::WEAR::WearType wtype = CORE::UTILS::IntegralValue<INPAR::WEAR::WearType>(
      GLOBAL::Problem::Instance()->WearParams(), "WEARTYPE");

  // get info about wear coeff conf
  INPAR::WEAR::WearCoeffConf wcoeffconf = CORE::UTILS::IntegralValue<INPAR::WEAR::WearCoeffConf>(
      GLOBAL::Problem::Instance()->WearParams(), "WEARCOEFF_CONF");

  if (interfaces_.size() > 1)
    FOUR_C_THROW("Wear algorithm not able to handle more than 1 interface yet!");

  //------------------------------------------------
  // Wear coefficient constant in material config: -
  //------------------------------------------------
  if (wcoeffconf == INPAR::WEAR::wear_coeff_mat)
  {
    // redistribute int. according to spatial interfaces!
    RedistributeMatInterfaces();

    // 1. pull back slave wear to material conf.
    WearPullBackSlave(disinterface_s);

    // 2. pull back master wear to material conf.
    if (wside == INPAR::WEAR::wear_both)
    {
      WearPullBackMaster(disinterface_m);
    }
    else
    {
      // zeroes
      Teuchos::RCP<Epetra_Map> masterdofs = interfaces_[0]->MasterRowDofs();
      disinterface_m = Teuchos::rcp(new Epetra_Vector(*masterdofs, true));
    }
  }
  //------------------------------------------------
  // Wear coefficient constant in spatial config:  -
  //------------------------------------------------
  else if (wcoeffconf == INPAR::WEAR::wear_coeff_sp)
  {
    // postproc wear for spatial conf.
    WearSpatialSlave(disinterface_s);

    if (wside == INPAR::WEAR::wear_both and wtype == INPAR::WEAR::wear_primvar)
    {
      WearSpatialMaster(disinterface_m);
    }
    else if (wside == INPAR::WEAR::wear_both and wtype == INPAR::WEAR::wear_intstate)
    {
      // redistribute int. according to spatial interfaces!
      RedistributeMatInterfaces();
      WearSpatialMasterMap(disinterface_s, disinterface_m);
    }
    else
    {
      // zeroes
      Teuchos::RCP<Epetra_Map> masterdofs = interfaces_[0]->MasterRowDofs();
      disinterface_m = Teuchos::rcp(new Epetra_Vector(*masterdofs, true));
    }
  }
  //------------------------------------------------
  // ERROR                                         -
  //------------------------------------------------
  else
  {
    FOUR_C_THROW("Chosen wear configuration not supported!");
  }

  return;
}


/*----------------------------------------------------------------------*
 | Wear in spatial conf.                                    farah 09/14 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::WearSpatialMasterMap(
    Teuchos::RCP<Epetra_Vector>& disinterface_s, Teuchos::RCP<Epetra_Vector>& disinterface_m)
{
  if (disinterface_s == Teuchos::null) FOUR_C_THROW("no slave wear for mapping!");

  // stactic cast of mortar strategy to contact strategy
  MORTAR::StrategyBase& strategy = cmtman_->GetStrategy();
  WEAR::LagrangeStrategyWear& cstrategy = static_cast<WEAR::LagrangeStrategyWear&>(strategy);

  for (int i = 0; i < (int)interfaces_.size(); ++i)
  {
    Teuchos::RCP<WEAR::WearInterface> winterface =
        Teuchos::rcp_dynamic_cast<WEAR::WearInterface>(interfacesMat_[i]);
    if (winterface == Teuchos::null) FOUR_C_THROW("Casting to WearInterface returned null!");

    Teuchos::RCP<Epetra_Map> masterdofs = interfaces_[i]->MasterRowDofs();
    Teuchos::RCP<Epetra_Map> slavedofs = interfaces_[i]->SlaveRowDofs();
    Teuchos::RCP<Epetra_Map> activedofs = interfaces_[i]->ActiveDofs();

    disinterface_m = Teuchos::rcp(new Epetra_Vector(*masterdofs, true));

    // different wear coefficients on both sides...
    double wearcoeff_s = interfaces_[i]->InterfaceParams().get<double>("WEARCOEFF", 0.0);
    double wearcoeff_m = interfaces_[i]->InterfaceParams().get<double>("WEARCOEFF_MASTER", 0.0);
    if (wearcoeff_s < 1e-12) FOUR_C_THROW("wcoeff negative!!!");

    double fac = wearcoeff_m / (wearcoeff_s);

    Teuchos::RCP<Epetra_Vector> wear_master = Teuchos::rcp(new Epetra_Vector(*masterdofs, true));

    cstrategy.MMatrix()->Multiply(true, *disinterface_s, *wear_master);

    // 1. set state to material displacement state
    winterface->SetState(MORTAR::state_new_displacement, *StructureField()->WriteAccessDispnp());

    // 2. initialize
    winterface->Initialize();

    // 3. calc N and areas
    winterface->SetElementAreas();
    winterface->EvaluateNodalNormals();

    // 6. init data container for d2 mat
    const Teuchos::RCP<Epetra_Map> masternodesmat =
        CORE::LINALG::AllreduceEMap(*(winterface->MasterRowNodes()));

    for (int i = 0; i < masternodesmat->NumMyElements();
         ++i)  // for (int i=0;i<MasterRowNodes()->NumMyElements();++i)
    {
      int gid = masternodesmat->GID(i);
      DRT::Node* node = winterface->Discret().gNode(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

      if (cnode->IsSlave() == false)
      {
        // reset nodal Mortar maps
        for (int j = 0; j < (int)((cnode->WearData().GetD2()).size()); ++j)
          (cnode->WearData().GetD2())[j].clear();

        (cnode->WearData().GetD2()).resize(0);
      }
    }

    // 8. evaluate dmat
    Teuchos::RCP<CORE::LINALG::SparseMatrix> dmat = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
        *masterdofs, 100, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));

    for (int j = 0; j < winterface->MasterColElements()->NumMyElements(); ++j)
    {
      int gid = winterface->MasterColElements()->GID(j);
      DRT::Element* ele = winterface->Discret().gElement(gid);
      if (!ele) FOUR_C_THROW("Cannot find ele with gid %", gid);
      CONTACT::Element* cele = dynamic_cast<CONTACT::Element*>(ele);

      Teuchos::RCP<CONTACT::Integrator> integrator = Teuchos::rcp(
          new CONTACT::Integrator(winterface->InterfaceParams(), cele->Shape(), Comm()));

      integrator->IntegrateD(*cele, Comm());
    }

    // 10. assemble dmat
    winterface->AssembleD2(*dmat);

    // 12. complete dmat
    dmat->Complete();

    Teuchos::ParameterList solvparams;
    CORE::UTILS::AddEnumClassToParameterList<CORE::LINEAR_SOLVER::SolverType>(
        "SOLVER", CORE::LINEAR_SOLVER::SolverType::umfpack, solvparams);
    CORE::LINALG::Solver solver(solvparams, Comm());

    CORE::LINALG::SolverParams solver_params;
    solver_params.refactor = true;
    solver.Solve(dmat->EpetraMatrix(), disinterface_m, wear_master, solver_params);

    disinterface_m->Scale(-fac);
  }

  // bye
  return;
}


/*----------------------------------------------------------------------*
 | Wear in spatial conf.                                    farah 09/14 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::WearSpatialMaster(Teuchos::RCP<Epetra_Vector>& disinterface_m)
{
  // get info about wear conf
  INPAR::WEAR::WearTimeScale wtime = CORE::UTILS::IntegralValue<INPAR::WEAR::WearTimeScale>(
      GLOBAL::Problem::Instance()->WearParams(), "WEAR_TIMESCALE");

  for (int i = 0; i < (int)interfaces_.size(); ++i)
  {
    Teuchos::RCP<Epetra_Map> masterdofs = interfaces_[i]->MasterRowDofs();
    disinterface_m = Teuchos::rcp(new Epetra_Vector(*masterdofs, true));

    // FIRST: get the wear values and the normal directions for the interface
    // loop over all slave row nodes on the current interface
    for (int j = 0; j < interfaces_[i]->MasterRowNodes()->NumMyElements(); ++j)
    {
      int gid = interfaces_[i]->MasterRowNodes()->GID(j);
      DRT::Node* node = interfaces_[i]->Discret().gNode(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

      // be aware of problem dimension
      int numdof = frinode->NumDof();
      if (dim_ != numdof) FOUR_C_THROW("Inconsistency Dim <-> NumDof");

      // nodal normal vector and wear
      double nn[3];
      double wear = 0.0;

      for (int j = 0; j < 3; ++j) nn[j] = frinode->MoData().n()[j];

      if (wtime == INPAR::WEAR::wear_time_different)
      {
        if (abs(frinode->WearData().wcurr()[0] + frinode->WearData().waccu()[0]) > 1e-12)
          wear = frinode->WearData().wcurr()[0] + frinode->WearData().waccu()[0];
        else
          wear = 0.0;
      }
      else
      {
        if (abs(frinode->WearData().wcurr()[0]) > 1e-12)
          wear = frinode->WearData().wcurr()[0];
        else
          wear = 0.0;
      }


      // find indices for DOFs of current node in Epetra_Vector
      // and put node values (normal and tangential stress components) at these DOFs
      std::vector<int> locindex(dim_);

      for (int dof = 0; dof < dim_; ++dof)
      {
        locindex[dof] = (disinterface_m->Map()).LID(frinode->Dofs()[dof]);
        (*disinterface_m)[locindex[dof]] = -wear * nn[dof];
      }
    }
  }

  // bye
  return;
}


/*----------------------------------------------------------------------*
 | Wear in spatial conf.                                    farah 09/14 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::WearSpatialSlave(Teuchos::RCP<Epetra_Vector>& disinterface_s)
{
  // stactic cast of mortar strategy to contact strategy
  MORTAR::StrategyBase& strategy = cmtman_->GetStrategy();
  WEAR::LagrangeStrategyWear& cstrategy = static_cast<WEAR::LagrangeStrategyWear&>(strategy);

  INPAR::WEAR::WearType wtype = CORE::UTILS::IntegralValue<INPAR::WEAR::WearType>(
      GLOBAL::Problem::Instance()->WearParams(), "WEARTYPE");

  INPAR::WEAR::WearTimInt wtimint = CORE::UTILS::IntegralValue<INPAR::WEAR::WearTimInt>(
      GLOBAL::Problem::Instance()->WearParams(), "WEARTIMINT");

  INPAR::WEAR::WearTimeScale wtime = CORE::UTILS::IntegralValue<INPAR::WEAR::WearTimeScale>(
      GLOBAL::Problem::Instance()->WearParams(), "WEAR_TIMESCALE");

  if (!(wtype == INPAR::WEAR::wear_intstate and wtimint == INPAR::WEAR::wear_impl))
    cstrategy.StoreNodalQuantities(MORTAR::StrategyBase::weightedwear);

  for (int i = 0; i < (int)interfaces_.size(); ++i)
  {
    Teuchos::RCP<Epetra_Map> slavedofs = interfaces_[i]->SlaveRowDofs();
    Teuchos::RCP<Epetra_Map> activedofs = interfaces_[i]->ActiveDofs();

    // additional spatial displacements
    disinterface_s = Teuchos::rcp(new Epetra_Vector(*slavedofs, true));

    // FIRST: get the wear values and the normal directions for the interface
    // loop over all slave row nodes on the current interface
    for (int j = 0; j < interfaces_[i]->SlaveRowNodes()->NumMyElements(); ++j)
    {
      int gid = interfaces_[i]->SlaveRowNodes()->GID(j);
      DRT::Node* node = interfaces_[i]->Discret().gNode(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

      // be aware of problem dimension
      int numdof = frinode->NumDof();
      if (dim_ != numdof) FOUR_C_THROW("Inconsistency Dim <-> NumDof");

      // nodal normal vector and wear
      double nn[3];
      double wear = 0.0;

      for (int j = 0; j < 3; ++j) nn[j] = frinode->MoData().n()[j];

      if (wtype == INPAR::WEAR::wear_primvar)
      {
        if (wtime == INPAR::WEAR::wear_time_different)
        {
          if (abs(frinode->WearData().wcurr()[0] + frinode->WearData().waccu()[0]) > 1e-12)
            wear = frinode->WearData().wcurr()[0] + frinode->WearData().waccu()[0];
          else
            wear = 0.0;
        }
        else
        {
          if (abs(frinode->WearData().wcurr()[0]) > 1e-12)
            wear = frinode->WearData().wcurr()[0];
          else
            wear = 0.0;
        }
      }
      else if (wtype == INPAR::WEAR::wear_intstate)
      {
        wear = frinode->WearData().WeightedWear();
      }

      // find indices for DOFs of current node in Epetra_Vector
      // and put node values (normal and tangential stress components) at these DOFs
      std::vector<int> locindex(dim_);

      for (int dof = 0; dof < dim_; ++dof)
      {
        locindex[dof] = (disinterface_s->Map()).LID(frinode->Dofs()[dof]);
        (*disinterface_s)[locindex[dof]] = -wear * nn[dof];
      }
    }

    // un-weight for internal state approach
    if (wtype == INPAR::WEAR::wear_intstate)
    {
      Teuchos::RCP<CORE::LINALG::SparseMatrix> daa, dai, dia, dii;
      Teuchos::RCP<Epetra_Map> gidofs;
      CORE::LINALG::SplitMatrix2x2(
          cstrategy.DMatrix(), activedofs, gidofs, activedofs, gidofs, daa, dai, dia, dii);

      Teuchos::RCP<Epetra_Vector> wear_vectora = Teuchos::rcp(new Epetra_Vector(*activedofs, true));
      Teuchos::RCP<Epetra_Vector> wear_vectori = Teuchos::rcp(new Epetra_Vector(*gidofs));
      CORE::LINALG::SplitVector(
          *slavedofs, *disinterface_s, activedofs, wear_vectora, gidofs, wear_vectori);

      Teuchos::RCP<Epetra_Vector> zref = Teuchos::rcp(new Epetra_Vector(*activedofs));

      // solve with default solver
      Teuchos::ParameterList solvparams;
      CORE::UTILS::AddEnumClassToParameterList<CORE::LINEAR_SOLVER::SolverType>(
          "SOLVER", CORE::LINEAR_SOLVER::SolverType::umfpack, solvparams);
      CORE::LINALG::Solver solver(solvparams, Comm());

      if (activedofs->NumMyElements())
      {
        CORE::LINALG::SolverParams solver_params;
        solver_params.refactor = true;

        solver.Solve(daa->EpetraMatrix(), zref, wear_vectora, solver_params);
      }

      // different wear coefficients on both sides...
      double wearcoeff_s = interfaces_[0]->InterfaceParams().get<double>("WEARCOEFF", 0.0);
      double wearcoeff_m = interfaces_[0]->InterfaceParams().get<double>("WEARCOEFF_MASTER", 0.0);
      if (wearcoeff_s < 1e-12) FOUR_C_THROW("wcoeff negative!!!");
      double fac = wearcoeff_s / (wearcoeff_s + wearcoeff_m);
      zref->Scale(fac);

      disinterface_s = Teuchos::rcp(new Epetra_Vector(*slavedofs));
      CORE::LINALG::Export(*zref, *disinterface_s);
    }
  }

  // bye
  return;
}


/*----------------------------------------------------------------------*
 | Redistribute material interfaces acc. to cur interf.     farah 09/14 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::RedistributeMatInterfaces()
{
  // barrier
  Comm().Barrier();

  // loop over all interfaces
  for (int m = 0; m < (int)interfaces_.size(); ++m)
  {
    int redistglobal = 0;
    int redistlocal = 0;
    if (interfaces_[m]->IsRedistributed()) redistlocal++;

    Comm().SumAll(&redistlocal, &redistglobal, 1);
    Comm().Barrier();


    if (redistglobal > 0)
    {
      if (Comm().MyPID() == 0)
      {
        std::cout << "===========================================" << std::endl;
        std::cout << "=======    Redistribute Mat. Int.   =======" << std::endl;
        std::cout << "===========================================" << std::endl;
      }
      Teuchos::RCP<WEAR::WearInterface> winterface =
          Teuchos::rcp_dynamic_cast<WEAR::WearInterface>(interfacesMat_[m]);

      // export nodes and elements to the row map
      winterface->Discret().ExportRowNodes(*interfaces_[m]->Discret().NodeRowMap());
      winterface->Discret().ExportRowElements(*interfaces_[m]->Discret().ElementRowMap());

      // export nodes and elements to the column map (create ghosting)
      winterface->Discret().ExportColumnNodes(*interfaces_[m]->Discret().NodeColMap());
      winterface->Discret().ExportColumnElements(*interfaces_[m]->Discret().ElementColMap());

      winterface->FillComplete(true);
      winterface->PrintParallelDistribution();

      if (Comm().MyPID() == 0)
      {
        std::cout << "===========================================" << std::endl;
        std::cout << "==============     Done!     ==============" << std::endl;
        std::cout << "===========================================" << std::endl;
      }
    }
  }

  // barrier
  Comm().Barrier();
  return;
}

/*----------------------------------------------------------------------*
 | Pull-Back wear: W = w * ds/dS * N                        farah 09/14 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::WearPullBackSlave(Teuchos::RCP<Epetra_Vector>& disinterface_s)
{
  // stactic cast of mortar strategy to contact strategy
  MORTAR::StrategyBase& strategy = cmtman_->GetStrategy();
  WEAR::LagrangeStrategyWear& cstrategy = dynamic_cast<WEAR::LagrangeStrategyWear&>(strategy);

  INPAR::WEAR::WearType wtype = CORE::UTILS::IntegralValue<INPAR::WEAR::WearType>(
      GLOBAL::Problem::Instance()->WearParams(), "WEARTYPE");

  INPAR::WEAR::WearTimInt wtimint = CORE::UTILS::IntegralValue<INPAR::WEAR::WearTimInt>(
      GLOBAL::Problem::Instance()->WearParams(), "WEARTIMINT");

  INPAR::WEAR::WearTimeScale wtime = CORE::UTILS::IntegralValue<INPAR::WEAR::WearTimeScale>(
      GLOBAL::Problem::Instance()->WearParams(), "WEAR_TIMESCALE");

  if (!(wtype == INPAR::WEAR::wear_intstate and wtimint == INPAR::WEAR::wear_impl))
    cstrategy.StoreNodalQuantities(MORTAR::StrategyBase::weightedwear);

  // loop over all interfaces
  for (int m = 0; m < (int)interfaces_.size(); ++m)
  {
    Teuchos::RCP<WEAR::WearInterface> winterface =
        Teuchos::rcp_dynamic_cast<WEAR::WearInterface>(interfaces_[m]);
    if (winterface == Teuchos::null) FOUR_C_THROW("Casting to WearInterface returned null!");

    // get slave row dofs as map
    Teuchos::RCP<Epetra_Map> slavedofs = winterface->SlaveRowDofs();
    // additional spatial displacements
    disinterface_s = Teuchos::rcp(new Epetra_Vector(*slavedofs, true));

    // call material interfaces and evaluate!
    // 1. set state to material displacement state
    interfacesMat_[m]->SetState(MORTAR::state_new_displacement, *StructureField()->DispMat());

    // 2. initialize
    interfacesMat_[m]->Initialize();

    // 3. calc N and areas
    interfacesMat_[m]->SetElementAreas();
    interfacesMat_[m]->EvaluateNodalNormals();

    // 4. calc -w*N
    for (int j = 0; j < winterface->SlaveRowNodes()->NumMyElements(); ++j)
    {
      int gid = winterface->SlaveRowNodes()->GID(j);
      DRT::Node* node = winterface->Discret().gNode(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

      int gidm = interfacesMat_[m]->SlaveRowNodes()->GID(j);
      DRT::Node* nodem = interfacesMat_[m]->Discret().gNode(gidm);
      if (!nodem) FOUR_C_THROW("Cannot find node with gid %", gidm);
      CONTACT::FriNode* frinodem = dynamic_cast<CONTACT::FriNode*>(nodem);

      // be aware of problem dimension
      int numdof = frinode->NumDof();
      if (dim_ != numdof) FOUR_C_THROW("Inconsistency Dim <-> NumDof");

      // nodal normal vector and wear
      double nn[3];
      double wear = 0.0;

      // get material normal
      for (int j = 0; j < 3; ++j) nn[j] = frinodem->MoData().n()[j];

      // primary variable approach:
      if (wtype == INPAR::WEAR::wear_primvar)
      {
        if (wtime == INPAR::WEAR::wear_time_different)
        {
          if (abs(frinode->WearData().wcurr()[0] + frinode->WearData().waccu()[0]) > 1e-12)
            wear = frinode->WearData().wcurr()[0] + frinode->WearData().waccu()[0];
          else
            wear = 0.0;
        }
        else
        {
          if (abs(frinode->WearData().wcurr()[0]) > 1e-12)
            wear = frinode->WearData().wcurr()[0];
          else
            wear = 0.0;
        }
      }
      // internal state variable approach:
      else if (wtype == INPAR::WEAR::wear_intstate)
      {
        wear = frinode->WearData().WeightedWear();
      }

      // find indices for DOFs of current node in Epetra_Vector
      // and put node values (normal and tangential stress components) at these DOFs
      std::vector<int> locindex(dim_);

      for (int dof = 0; dof < dim_; ++dof)
      {
        locindex[dof] = (disinterface_s->Map()).LID(frinode->Dofs()[dof]);
        (*disinterface_s)[locindex[dof]] = -wear * nn[dof];
      }
    }

    // 5. evaluate dmat
    Teuchos::RCP<CORE::LINALG::SparseMatrix> dmat =
        Teuchos::rcp(new CORE::LINALG::SparseMatrix(*slavedofs, 10));

    for (int j = 0; j < interfacesMat_[m]->SlaveColElements()->NumMyElements(); ++j)
    {
      int gid = interfacesMat_[m]->SlaveColElements()->GID(j);
      DRT::Element* ele = interfacesMat_[m]->Discret().gElement(gid);
      if (!ele) FOUR_C_THROW("Cannot find ele with gid %", gid);
      CONTACT::Element* cele = dynamic_cast<CONTACT::Element*>(ele);

      Teuchos::RCP<CONTACT::Integrator> integrator = Teuchos::rcp(
          new CONTACT::Integrator(interfacesMat_[m]->InterfaceParams(), cele->Shape(), Comm()));

      integrator->IntegrateD(*cele, Comm());
    }

    // 6. assemble dmat
    interfacesMat_[m]->AssembleD(*dmat);

    // 7. complete dmat
    dmat->Complete();

    // 8. area trafo:
    if (wtype == INPAR::WEAR::wear_primvar)
    {
      // multiply current D matrix with current wear
      Teuchos::RCP<Epetra_Vector> forcecurr = Teuchos::rcp(new Epetra_Vector(*slavedofs));
      cstrategy.DMatrix()->Multiply(false, *disinterface_s, *forcecurr);

      // LM in reference / current configuration
      Teuchos::RCP<Epetra_Vector> zref = Teuchos::rcp(new Epetra_Vector(*slavedofs));

      // solve with default solver
      Teuchos::ParameterList solvparams;
      CORE::UTILS::AddEnumClassToParameterList<CORE::LINEAR_SOLVER::SolverType>(
          "SOLVER", CORE::LINEAR_SOLVER::SolverType::umfpack, solvparams);
      CORE::LINALG::Solver solver(solvparams, Comm());

      CORE::LINALG::SolverParams solver_params;
      solver_params.refactor = true;
      solver.Solve(dmat->EpetraOperator(), zref, forcecurr, solver_params);


      // store reference LM into global vector and nodes
      disinterface_s = zref;
    }
    else if (wtype == INPAR::WEAR::wear_intstate)
    {
      Teuchos::RCP<Epetra_Vector> zref = Teuchos::rcp(new Epetra_Vector(*slavedofs));

      // solve with default solver
      Teuchos::ParameterList solvparams;
      CORE::UTILS::AddEnumClassToParameterList<CORE::LINEAR_SOLVER::SolverType>(
          "SOLVER", CORE::LINEAR_SOLVER::SolverType::umfpack, solvparams);
      CORE::LINALG::Solver solver(solvparams, Comm());

      CORE::LINALG::SolverParams solver_params;
      solver_params.refactor = true;
      solver.Solve(dmat->EpetraOperator(), zref, disinterface_s, solver_params);


      // store reference LM into global vector and nodes
      disinterface_s = zref;

      // different wear coefficients on both sides...
      double wearcoeff_s = interfaces_[0]->InterfaceParams().get<double>("WEARCOEFF", 0.0);
      double wearcoeff_m = interfaces_[0]->InterfaceParams().get<double>("WEARCOEFF_MASTER", 0.0);
      if (wearcoeff_s < 1e-12) FOUR_C_THROW("wcoeff negative!!!");

      double fac = wearcoeff_s / (wearcoeff_s + wearcoeff_m);
      disinterface_s->Scale(fac);
    }
    else
    {
      FOUR_C_THROW("wrong wear type!");
    }
  }

  // bye
  return;
}


/*----------------------------------------------------------------------*
 | Pull-Back wear: W = w * ds/dS * N                        farah 09/14 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::WearPullBackMaster(Teuchos::RCP<Epetra_Vector>& disinterface_m)
{
  INPAR::WEAR::WearType wtype = CORE::UTILS::IntegralValue<INPAR::WEAR::WearType>(
      GLOBAL::Problem::Instance()->WearParams(), "WEARTYPE");

  INPAR::WEAR::WearTimeScale wtime = CORE::UTILS::IntegralValue<INPAR::WEAR::WearTimeScale>(
      GLOBAL::Problem::Instance()->WearParams(), "WEAR_TIMESCALE");

  // loop over all interfaces
  for (int m = 0; m < (int)interfaces_.size(); ++m)
  {
    Teuchos::RCP<WEAR::WearInterface> winterface =
        Teuchos::rcp_dynamic_cast<WEAR::WearInterface>(interfaces_[m]);
    if (winterface == Teuchos::null) FOUR_C_THROW("Casting to WearInterface returned null!");

    Teuchos::RCP<WEAR::WearInterface> winterfaceMat =
        Teuchos::rcp_dynamic_cast<WEAR::WearInterface>(interfacesMat_[m]);
    if (winterfaceMat == Teuchos::null) FOUR_C_THROW("Casting to WearInterface returned null!");

    // get slave row dofs as map
    Teuchos::RCP<Epetra_Map> masterdofs = winterface->MasterRowDofs();
    // additional spatial displacements
    disinterface_m = Teuchos::rcp(new Epetra_Vector(*masterdofs, true));

    // call material interfaces and evaluate!
    // 1. set state to material displacement state
    winterfaceMat->SetState(MORTAR::state_new_displacement, *StructureField()->DispMat());

    // 2. initialize
    winterfaceMat->Initialize();

    // 3. calc N and areas
    winterfaceMat->SetElementAreas();
    winterfaceMat->EvaluateNodalNormals();

    // 4. calc -w*N
    for (int j = 0; j < winterface->MasterRowNodes()->NumMyElements(); ++j)
    {
      int gid = winterface->MasterRowNodes()->GID(j);
      DRT::Node* node = winterface->Discret().gNode(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* frinode = dynamic_cast<CONTACT::FriNode*>(node);

      int gidm = interfacesMat_[m]->MasterRowNodes()->GID(j);
      DRT::Node* nodem = interfacesMat_[m]->Discret().gNode(gidm);
      if (!nodem) FOUR_C_THROW("Cannot find node with gid %", gidm);
      CONTACT::FriNode* frinodem = dynamic_cast<CONTACT::FriNode*>(nodem);

      // be aware of problem dimension
      int numdof = frinode->NumDof();
      if (dim_ != numdof) FOUR_C_THROW("Inconsistency Dim <-> NumDof");

      // nodal normal vector and wear
      double nn[3];
      double wear = 0.0;

      // get material normal
      for (int j = 0; j < 3; ++j) nn[j] = frinodem->MoData().n()[j];

      if (wtype == INPAR::WEAR::wear_primvar)
      {
        if (wtime == INPAR::WEAR::wear_time_different)
        {
          if (abs(frinode->WearData().wcurr()[0] + frinode->WearData().waccu()[0]) > 1e-12)
            wear = frinode->WearData().wcurr()[0] + frinode->WearData().waccu()[0];
          else
            wear = 0.0;
        }
        else
        {
          if (abs(frinode->WearData().wcurr()[0]) > 1e-12)
            wear = frinode->WearData().wcurr()[0];
          else
            wear = 0.0;
        }
      }
      else if (wtype == INPAR::WEAR::wear_intstate)
      {
        wear = frinode->WearData().WeightedWear();
      }

      // find indices for DOFs of current node in Epetra_Vector
      // and put node values (normal and tangential stress components) at these DOFs
      std::vector<int> locindex(dim_);

      for (int dof = 0; dof < dim_; ++dof)
      {
        locindex[dof] = (disinterface_m->Map()).LID(frinode->Dofs()[dof]);
        (*disinterface_m)[locindex[dof]] = -wear * nn[dof];
      }
    }

    // 5. init data container for d2 curr
    const Teuchos::RCP<Epetra_Map> masternodes =
        CORE::LINALG::AllreduceEMap(*(winterface->MasterRowNodes()));

    for (int i = 0; i < masternodes->NumMyElements();
         ++i)  // for (int i=0;i<MasterRowNodes()->NumMyElements();++i)
    {
      int gid = masternodes->GID(i);
      DRT::Node* node = winterface->Discret().gNode(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

      if (cnode->IsSlave() == false)
      {
        // reset nodal Mortar maps
        for (int j = 0; j < (int)((cnode->WearData().GetD2()).size()); ++j)
          (cnode->WearData().GetD2())[j].clear();

        (cnode->WearData().GetD2()).resize(0);
      }
    }

    // 6. init data container for d2 mat
    const Teuchos::RCP<Epetra_Map> masternodesmat =
        CORE::LINALG::AllreduceEMap(*(winterfaceMat->MasterRowNodes()));

    for (int i = 0; i < masternodesmat->NumMyElements();
         ++i)  // for (int i=0;i<MasterRowNodes()->NumMyElements();++i)
    {
      int gid = masternodesmat->GID(i);
      DRT::Node* node = winterfaceMat->Discret().gNode(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      CONTACT::FriNode* cnode = dynamic_cast<CONTACT::FriNode*>(node);

      if (cnode->IsSlave() == false)
      {
        // reset nodal Mortar maps
        for (int j = 0; j < (int)((cnode->WearData().GetD2()).size()); ++j)
          (cnode->WearData().GetD2())[j].clear();

        (cnode->WearData().GetD2()).resize(0);
      }
    }

    // 7. evaluate dcur
    Teuchos::RCP<CORE::LINALG::SparseMatrix> dcur = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
        *masterdofs, 100, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));
    for (int j = 0; j < winterface->MasterColElements()->NumMyElements(); ++j)
    {
      int gid = winterface->MasterColElements()->GID(j);
      DRT::Element* ele = winterface->Discret().gElement(gid);
      if (!ele) FOUR_C_THROW("Cannot find ele with gid %", gid);
      CONTACT::Element* cele = dynamic_cast<CONTACT::Element*>(ele);

      Teuchos::RCP<CONTACT::Integrator> integrator = Teuchos::rcp(
          new CONTACT::Integrator(winterface->InterfaceParams(), cele->Shape(), Comm()));

      integrator->IntegrateD(*cele, Comm());
    }

    // 8. evaluate dmat
    Teuchos::RCP<CORE::LINALG::SparseMatrix> dmat = Teuchos::rcp(new CORE::LINALG::SparseMatrix(
        *masterdofs, 100, true, false, CORE::LINALG::SparseMatrix::FE_MATRIX));

    for (int j = 0; j < winterfaceMat->MasterColElements()->NumMyElements(); ++j)
    {
      int gid = winterfaceMat->MasterColElements()->GID(j);
      DRT::Element* ele = winterfaceMat->Discret().gElement(gid);
      if (!ele) FOUR_C_THROW("Cannot find ele with gid %", gid);
      CONTACT::Element* cele = dynamic_cast<CONTACT::Element*>(ele);

      Teuchos::RCP<CONTACT::Integrator> integrator = Teuchos::rcp(
          new CONTACT::Integrator(winterfaceMat->InterfaceParams(), cele->Shape(), Comm()));

      integrator->IntegrateD(*cele, Comm());
    }

    // 9. assemble dcur
    winterface->AssembleD2(*dcur);

    // 10. assemble dmat
    winterfaceMat->AssembleD2(*dmat);

    // 11. complete dcur
    dcur->Complete();

    // 12. complete dmat
    dmat->Complete();

    // 13. area trafo:
    if (wtype == INPAR::WEAR::wear_primvar)
    {
      // multiply current D matrix with current wear
      Teuchos::RCP<Epetra_Vector> forcecurr = Teuchos::rcp(new Epetra_Vector(*masterdofs));
      dcur->Multiply(false, *disinterface_m, *forcecurr);

      // LM in reference / current configuration
      Teuchos::RCP<Epetra_Vector> zref = Teuchos::rcp(new Epetra_Vector(*masterdofs));

      // solve with default solver
      Teuchos::ParameterList solvparams;
      CORE::UTILS::AddEnumClassToParameterList<CORE::LINEAR_SOLVER::SolverType>(
          "SOLVER", CORE::LINEAR_SOLVER::SolverType::umfpack, solvparams);
      CORE::LINALG::Solver solver(solvparams, Comm());

      CORE::LINALG::SolverParams solver_params;
      solver_params.refactor = true;
      solver.Solve(dmat->EpetraOperator(), zref, forcecurr, solver_params);


      // store reference LM into global vector and nodes
      disinterface_m = zref;
    }
    else if (wtype == INPAR::WEAR::wear_intstate)
    {
      FOUR_C_THROW("not working yet!");
      Teuchos::RCP<Epetra_Vector> zref = Teuchos::rcp(new Epetra_Vector(*masterdofs));

      // solve with default solver
      Teuchos::ParameterList solvparams;
      CORE::UTILS::AddEnumClassToParameterList<CORE::LINEAR_SOLVER::SolverType>(
          "SOLVER", CORE::LINEAR_SOLVER::SolverType::umfpack, solvparams);
      CORE::LINALG::Solver solver(solvparams, Comm());

      CORE::LINALG::SolverParams solver_params;
      solver_params.refactor = true;
      solver.Solve(dmat->EpetraOperator(), zref, disinterface_m, solver_params);

      // store reference LM into global vector and nodes
      disinterface_m = zref;
    }
    else
    {
      FOUR_C_THROW("wrong wear type!");
    }
  }

  // bye
  return;
}


/*----------------------------------------------------------------------*
 | Application of mesh displacement to material conf         farah 04/15|
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::UpdateMatConf()
{
  // mesh displacement from solution of ALE field in structural dofs
  // first perform transformation from ale to structure dofs
  Teuchos::RCP<Epetra_Vector> disalenp = AleToStructure(AleField().Dispnp());

  // vector of current spatial displacements
  Teuchos::RCP<const Epetra_Vector> dispnp =
      StructureField()->Dispnp();  // change to ExtractDispn() for overlap

  // material displacements
  Teuchos::RCP<Epetra_Vector> dismat = Teuchos::rcp(new Epetra_Vector(dispnp->Map()), true);

  // set state
  (StructureField()->Discretization())->SetState(0, "displacement", dispnp);

  // set state
  (StructureField()->Discretization())
      ->SetState(0, "material_displacement", StructureField()->DispMat());

  // get info about wear conf
  INPAR::WEAR::WearShapeEvo wconf = CORE::UTILS::IntegralValue<INPAR::WEAR::WearShapeEvo>(
      GLOBAL::Problem::Instance()->WearParams(), "WEAR_SHAPE_EVO");

  // if shape evol. in mat conf: ale dispnp = material displ.
  if (wconf == INPAR::WEAR::wear_se_mat)
  {
    // just information for user
    int err = 0;
    err = delta_ale_->Update(-1.0, *ale_i_, 0.0);
    if (err != 0) FOUR_C_THROW("update wrong!");
    err = delta_ale_->Update(1.0, *AleField().Dispnp(), 1.0);
    if (err != 0) FOUR_C_THROW("update wrong!");
    err = ale_i_->Update(1.0, *AleField().Dispnp(), 0.0);
    if (err != 0) FOUR_C_THROW("update wrong!");

    // important vector to update mat conf
    Teuchos::RCP<Epetra_Vector> dismat_struct =
        Teuchos::rcp(new Epetra_Vector(dispnp->Map()), true);

    CORE::LINALG::Export(*disalenp, *dismat_struct);

    err = dismat->Update(1.0, *dismat_struct, 0.0);
    if (err != 0) FOUR_C_THROW("update wrong!");
  }
  // if shape evol. in spat conf: advection map!
  else if (wconf == INPAR::WEAR::wear_se_sp)
  {
    int err = 0;
    err = disalenp->Update(-1.0, *dispnp, 1.0);
    if (err != 0) FOUR_C_THROW("update wrong!");
    err = delta_ale_->Update(1.0, *StructureToAle(disalenp), 0.0);
    if (err != 0) FOUR_C_THROW("update wrong!");

    // loop over all row nodes to fill graph
    for (int k = 0; k < StructureField()->Discretization()->NumMyRowNodes(); ++k)
    {
      int gid = StructureField()->Discretization()->NodeRowMap()->GID(k);
      DRT::Node* node = StructureField()->Discretization()->gNode(gid);
      DRT::Element** ElementPtr = node->Elements();
      int numelement = node->NumElement();

      const int numdof = StructureField()->Discretization()->NumDof(node);

      // create Xmat for 3D problems
      std::vector<double> XMat(numdof);
      std::vector<double> XMesh(numdof);

      for (int dof = 0; dof < numdof; ++dof)
      {
        int dofgid = StructureField()->Discretization()->Dof(node, dof);
        int doflid = (dispnp->Map()).LID(dofgid);
        XMesh[dof] = node->X()[dof] + (*dispnp)[doflid] + (*disalenp)[doflid];
      }

      // create updated  XMat --> via nonlinear interpolation between nodes (like gp projection)
      AdvectionMap(XMat.data(), XMesh.data(), ElementPtr, numelement, true);

      // store in dispmat
      for (int dof = 0; dof < numdof; ++dof)
      {
        int dofgid = StructureField()->Discretization()->Dof(node, dof);
        int doflid = (dispnp->Map()).LID(dofgid);
        (*dismat)[doflid] = XMat[dof] - node->X()[dof];
      }
    }  // end row node loop
  }

  // apply material displacements to structural field
  // if advection map is not succesful --> use old xmat
  StructureField()->ApplyDisMat(dismat);

  // bye
  return;
}


/*----------------------------------------------------------------------*
 | material coordinates evaluated from spatial ones         farah 12/13 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::AdvectionMap(double* Xtarget,  // out
    double* Xsource,                                   // in
    DRT::Element** ElementPtr,                         // in
    int numelements,                                   // in
    bool spatialtomaterial)                            // in
{
  // get problem dimension
  const int ndim = GLOBAL::Problem::Instance()->NDim();

  // define source and target configuration
  std::string sourceconf;
  std::string targetconf;

  if (spatialtomaterial)
  {
    sourceconf = "displacement";
    targetconf = "material_displacement";
  }
  else
  {
    sourceconf = "material_displacement";
    targetconf = "displacement";
  }

  // found element the spatial coordinate lies in
  bool found = false;

  // parameter space coordinates
  double e[3];
  double ge1 = 1e12;
  double ge2 = 1e12;
  double ge3 = 1e12;
  int gele = 0;

  // get state
  Teuchos::RCP<const Epetra_Vector> dispsource =
      (StructureField()->Discretization())->GetState(sourceconf);
  Teuchos::RCP<const Epetra_Vector> disptarget =
      (StructureField()->Discretization())->GetState(targetconf);

  // loop over adjacent elements
  for (int jele = 0; jele < numelements; jele++)
  {
    // get element
    DRT::Element* actele = ElementPtr[jele];

    // get element location vector, dirichlet flags and ownerships
    DRT::Element::LocationArray la(1);
    actele->LocationVector(*(StructureField()->Discretization()), la, false);

    if (ndim == 2)
    {
      if (actele->Shape() == CORE::FE::CellType::quad4)
        WEAR::UTILS::av<CORE::FE::CellType::quad4>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else if (actele->Shape() == CORE::FE::CellType::quad8)
        WEAR::UTILS::av<CORE::FE::CellType::quad8>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else if (actele->Shape() == CORE::FE::CellType::quad9)
        WEAR::UTILS::av<CORE::FE::CellType::quad9>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else if (actele->Shape() == CORE::FE::CellType::tri3)
        WEAR::UTILS::av<CORE::FE::CellType::tri3>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else if (actele->Shape() == CORE::FE::CellType::tri6)
        WEAR::UTILS::av<CORE::FE::CellType::tri6>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else
        FOUR_C_THROW("shape function not supported!");

      // checks if the spatial coordinate lies within this element
      // if yes, returns the material displacements
      // w1ele->AdvectionMapElement(XMat1,XMat2,XMesh1,XMesh2,disp,dispmat, la,found,e1,e2);

      // if parameter space coord. 'e' does not lie within any element (i.e. found = false),
      // then jele is the element lying closest near the considered spatial point.
      if (found == false)
      {
        if (abs(ge1) > 1.0 and abs(e[0]) < abs(ge1))
        {
          ge1 = e[0];
          gele = jele;
        }
        if (abs(ge2) > 1.0 and abs(e[1]) < abs(ge2))
        {
          ge2 = e[1];
          gele = jele;
        }
      }
    }
    else
    {
      if (actele->ElementType() == DRT::ELEMENTS::SoHex8Type::Instance())
        WEAR::UTILS::av<CORE::FE::CellType::hex8>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else if (actele->ElementType() == DRT::ELEMENTS::SoHex20Type::Instance())
        WEAR::UTILS::av<CORE::FE::CellType::hex20>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else if (actele->ElementType() == DRT::ELEMENTS::SoHex27Type::Instance())
        WEAR::UTILS::av<CORE::FE::CellType::hex27>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else if (actele->ElementType() == DRT::ELEMENTS::SoTet4Type::Instance())
        WEAR::UTILS::av<CORE::FE::CellType::tet4>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else if (actele->ElementType() == DRT::ELEMENTS::SoTet10Type::Instance())
        WEAR::UTILS::av<CORE::FE::CellType::tet10>(
            actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
      else
        FOUR_C_THROW("element type not supported!");

      // if parameter space coord. 'e' does not lie within any element (i.e. found = false),
      // then 'gele = jele' is the element lying closest near the considered spatial point.
      if (found == false)
      {
        if (abs(ge1) > 1.0 and abs(e[0]) < abs(ge1))
        {
          ge1 = e[0];
          gele = jele;
        }
        if (abs(ge2) > 1.0 and abs(e[1]) < abs(ge2))
        {
          ge2 = e[1];
          gele = jele;
        }
        if (abs(ge3) > 1.0 and abs(e[2]) < abs(ge3))
        {
          ge3 = e[2];
          gele = jele;
        }
      }
    }

    // leave when element is found
    if (found == true) return;
  }  // end loop over adj elements

  // ****************************************
  //  if displ not into elements: get
  //  Xtarget from closest element 'gele'
  // ****************************************
  DRT::Element* actele = ElementPtr[gele];

  // get element location vector, dirichlet flags and ownerships
  DRT::Element::LocationArray la(1);
  actele->LocationVector(*(StructureField()->Discretization()), la, false);

  if (ndim == 2)
  {
    if (actele->Shape() == CORE::FE::CellType::quad4)
      WEAR::UTILS::av<CORE::FE::CellType::quad4>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else if (actele->Shape() == CORE::FE::CellType::quad8)
      WEAR::UTILS::av<CORE::FE::CellType::quad8>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else if (actele->Shape() == CORE::FE::CellType::quad9)
      WEAR::UTILS::av<CORE::FE::CellType::quad9>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else if (actele->Shape() == CORE::FE::CellType::tri3)
      WEAR::UTILS::av<CORE::FE::CellType::tri3>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else if (actele->Shape() == CORE::FE::CellType::tri6)
      WEAR::UTILS::av<CORE::FE::CellType::tri6>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else
      FOUR_C_THROW("shape function not supported!");
  }
  else
  {
    if (actele->ElementType() == DRT::ELEMENTS::SoHex8Type::Instance())
      WEAR::UTILS::av<CORE::FE::CellType::hex8>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else if (actele->ElementType() == DRT::ELEMENTS::SoHex20Type::Instance())
      WEAR::UTILS::av<CORE::FE::CellType::hex20>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else if (actele->ElementType() == DRT::ELEMENTS::SoHex27Type::Instance())
      WEAR::UTILS::av<CORE::FE::CellType::hex27>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else if (actele->ElementType() == DRT::ELEMENTS::SoTet4Type::Instance())
      WEAR::UTILS::av<CORE::FE::CellType::tet4>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else if (actele->ElementType() == DRT::ELEMENTS::SoTet10Type::Instance())
      WEAR::UTILS::av<CORE::FE::CellType::tet10>(
          actele, Xtarget, Xsource, dispsource, disptarget, la[0].lm_, found, e);
    else
      FOUR_C_THROW("element type not supported!");
  }

  // bye
  return;
}


/*----------------------------------------------------------------------*
 | Perform ALE step                                         farah 11/13 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::AleStep(Teuchos::RCP<Epetra_Vector> idisale_global)
{
  // get info about ale dynamic
  // INPAR::ALE::AleDynamic aletype =
  CORE::UTILS::IntegralValue<INPAR::ALE::AleDynamic>(ParamsAle(), "ALE_TYPE");

  // get info about wear conf
  INPAR::WEAR::WearShapeEvo wconf = CORE::UTILS::IntegralValue<INPAR::WEAR::WearShapeEvo>(
      GLOBAL::Problem::Instance()->WearParams(), "WEAR_SHAPE_EVO");

  //  if(aletype != INPAR::ALE::solid)
  //    FOUR_C_THROW("ERORR: Chosen ALE type not supported!");

  if (wconf == INPAR::WEAR::wear_se_sp)
  {
    //    Teuchos::RCP<Epetra_Vector> dispnpstru = StructureToAle(
    //        StructureField()->Dispnp());
    //
    //    FS3I::Biofilm::UTILS::updateMaterialConfigWithALE_Disp(
    //        AleField().WriteAccessDiscretization(),
    //        dispnpstru );
    //
    //    AleField().WriteAccessDispnp()->Update(0.0, *(dispnpstru), 0.0);
    //
    //    // application of interface displacements as dirichlet conditions
    //    //AleField().ApplyInterfaceDisplacements(idisale_global);
    //
    //    // solve time step
    //    AleField().TimeStep(ALE::UTILS::MapExtractor::dbc_set_wear);
    //
    //    AleField().WriteAccessDispnp()->Update(1.0, *(dispnpstru), 1.0);


    Teuchos::RCP<Epetra_Vector> dispnpstru = StructureToAle(StructureField()->Dispnp());

    AleField().WriteAccessDispnp()->Update(1.0, *(dispnpstru), 0.0);

    // application of interface displacements as dirichlet conditions
    AleField().ApplyInterfaceDisplacements(idisale_global);

    // solve time step
    AleField().TimeStep(ALE::UTILS::MapExtractor::dbc_set_wear);
  }
  // classical lin in mat. conf --> not correct at all
  else if (wconf == INPAR::WEAR::wear_se_mat)
  {
    // application of interface displacements as dirichlet conditions
    AleField().ApplyInterfaceDisplacements(idisale_global);

    // solve time step
    AleField().TimeStep(ALE::UTILS::MapExtractor::dbc_set_wear);
  }
  else
    FOUR_C_THROW("Chosen wear configuration not supported!");

  return;
}


/*----------------------------------------------------------------------*
 | transform from ale to structure map                      farah 11/13 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> WEAR::Partitioned::AleToStructure(Teuchos::RCP<Epetra_Vector> vec) const
{
  return coupalestru_->MasterToSlave(vec);
}


/*----------------------------------------------------------------------*
 | transform from ale to structure map                      farah 11/13 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> WEAR::Partitioned::AleToStructure(
    Teuchos::RCP<const Epetra_Vector> vec) const
{
  return coupalestru_->MasterToSlave(vec);
}


/*----------------------------------------------------------------------*
 | transform from ale to structure map                      farah 11/13 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> WEAR::Partitioned::StructureToAle(Teuchos::RCP<Epetra_Vector> vec) const
{
  return coupalestru_->SlaveToMaster(vec);
}


/*----------------------------------------------------------------------*
 | transform from ale to structure map                      farah 11/13 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> WEAR::Partitioned::StructureToAle(
    Teuchos::RCP<const Epetra_Vector> vec) const
{
  return coupalestru_->SlaveToMaster(vec);
}


/*----------------------------------------------------------------------*
 | read restart information for given time step (public)    farah 10/13 |
 *----------------------------------------------------------------------*/
void WEAR::Partitioned::ReadRestart(int step)
{
  StructureField()->ReadRestart(step);
  AleField().ReadRestart(step);
  SetTimeStep(StructureField()->TimeOld(), step);

  return;
}
/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE