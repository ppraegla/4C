/*----------------------------------------------------------------------*/
/*! \file

\brief Split conditions into map extractors

\level 1


*/
/*----------------------------------------------------------------------*/


#include "baci_lib_condition_selector.H"

#include "baci_lib_condition.H"
#include "baci_lib_discret.H"
#include "baci_linalg_utils_sparse_algebra_create.H"
#include "baci_utils_exceptions.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DRT::UTILS::ConditionSelector::ConditionSelector(
    const DRT::Discretization& dis, std::string condname)
    : dis_(dis)
{
  dis.GetCondition(condname, conds_);
  std::sort(conds_.begin(), conds_.end(), DRT::ConditionLess());
}


/*----------------------------------------------------------------------*
 | construct a selector from a given vector of conditions    fang 07/16 |
 *----------------------------------------------------------------------*/
DRT::UTILS::ConditionSelector::ConditionSelector(
    const DRT::Discretization& dis,            //!< discretization
    const std::vector<DRT::Condition*>& conds  //!< given vector of conditions
    )
    : dis_(dis), conds_(conds)
{
  std::sort(conds_.begin(), conds_.end(), DRT::ConditionLess());
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::UTILS::ConditionSelector::SelectDofs(DRT::Node* node, std::set<int>& conddofset)
{
  bool found = false;

  // put all conditioned dofs into conddofset
  if (ContainsNode(node->Id()))
  {
    std::vector<int> dof = Discretization().Dof(0, node);
    for (unsigned k = 0; k < dof.size(); ++k)
    {
      // test for dof position
      if (ContainsDof(dof[k], k))
      {
        conddofset.insert(dof[k]);
        found = true;
      }
    }
  }
  return found;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::UTILS::ConditionSelector::ContainsNode(int ngid)
{
  for (const auto& cond : conds_)
  {
    if (cond->ContainsNode(ngid))
    {
      return true;
    }
  }
  return false;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::UTILS::DirichletSelector::SelectDofs(DRT::Node* node, std::set<int>& conddofset)
{
  bool found = false;
  int ngid = node->Id();

  // The condition vector is sorted by condition type. Thus lesser entity
  // ranks are considered first. The first condition that covers a node gets
  // it.

  for (const auto& cond : Conditions())
  {
    const auto* onoff = cond->Get<std::vector<int>>("onoff");
    if (onoff == nullptr) dserror("not a valid Dirichlet condition");
    if (cond->ContainsNode(ngid))
    {
      std::vector<int> dof = Discretization().Dof(node);
      for (unsigned k = 0; k < dof.size(); ++k)
      {
        if (k > onoff->size()) dserror("not a valid Dirichlet condition");
        if ((*onoff)[k] != 0)
        {
          conddofset.insert(dof[k]);
        }
      }

      // if a node has been covered by one Dirichlet condition do not look for
      // further conditions
      found = true;
      break;
    }
  }
  return found;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DRT::UTILS::MultiConditionSelector::MultiConditionSelector() : overlapping_(false) {}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::MultiConditionSelector::SetupExtractor(const DRT::Discretization& dis,
    const Epetra_Map& fullmap, CORE::LINALG::MultiMapExtractor& extractor)
{
  SetupCondDofSets(dis);

  // Find all non-conditioned dofs by subtracting all conditioned ones.

  std::set<int> otherdofset(
      fullmap.MyGlobalElements(), fullmap.MyGlobalElements() + fullmap.NumMyElements());

  for (auto& conddofset : conddofset_)
  {
    for (const auto& dof : conddofset)
    {
      otherdofset.erase(dof);
    }
  }

  // Setup all maps. The "other" map goes first so it becomes the zeroth map
  // of the MultiMapExtractor.

  std::vector<Teuchos::RCP<const Epetra_Map>> maps;
  maps.reserve(conddofset_.size() + 1);

  maps.emplace_back(CORE::LINALG::CreateMap(otherdofset, dis.Comm()));
  for (auto& conddofset : conddofset_)
  {
    maps.emplace_back(CORE::LINALG::CreateMap(conddofset, dis.Comm()));
  }

  // MultiMapExtractor setup

  extractor.Setup(fullmap, maps);
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::MultiConditionSelector::SetupCondDofSets(const DRT::Discretization& dis)
{
  // we get as many sets as we have selectors
  conddofset_.resize(selectors_.size());

  // for each owned node
  int numrownodes = dis.NumMyRowNodes();
  for (int i = 0; i < numrownodes; ++i)
  {
    DRT::Node* node = dis.lRowNode(i);

    // test each selector
    for (unsigned j = 0; j < selectors_.size(); ++j)
    {
      ConditionSelector& conds = *selectors_[j];

      // if the selector applies, we are done
      if (conds.SelectDofs(node, conddofset_[j]))
        if (!overlapping_) break;
    }
  }
}