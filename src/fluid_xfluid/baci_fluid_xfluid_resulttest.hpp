/*----------------------------------------------------------------------*/
/*! \file

\brief xfem based fluid result tests

\level 0

 */
/*----------------------------------------------------------------------*/

#ifndef BACI_FLUID_XFLUID_RESULTTEST_HPP
#define BACI_FLUID_XFLUID_RESULTTEST_HPP


#include "baci_config.hpp"

#include "baci_lib_resulttest.hpp"

#include <Epetra_Vector.h>

BACI_NAMESPACE_OPEN

namespace DRT
{
  class Discretization;
}

namespace FLD
{
  // forward declarations
  class XFluid;
  class XFluidFluid;
  class XFluidFluid;

  /*!
    ResultTest class for XFluid
   */
  class XFluidResultTest : public DRT::ResultTest
  {
   public:
    //! ctor for standard XFEM problems
    XFluidResultTest(const FLD::XFluid& xfluid);

    //! ctor for XFF-problems
    XFluidResultTest(const FLD::XFluidFluid& xfluid);

    /// our version of nodal value tests
    /*!
      Possible position flags are "velx", "vely", "velz" and
      "pressure". With the obvious meaning.
     */
    void TestNode(INPUT::LineDefinition& res, int& nerr, int& test_count) override;

   private:
    /// nodal value test (one can specify discretization and corresponding solution here!)
    void TestNode(INPUT::LineDefinition& res, int& nerr, int& test_count, int node,
        const Teuchos::RCP<const DRT::Discretization>& discret,
        const Teuchos::RCP<const Epetra_Vector>& velnp);

    /// XFEM discretization
    Teuchos::RCP<const DRT::Discretization> discret_;

    /// solution vector for XFEM discretization
    Teuchos::RCP<const Epetra_Vector> velnp_;

    /// optional additional discretization for the same field (fluid-fluid coupling)
    Teuchos::RCP<const DRT::Discretization> coupl_discret_;

    /// solution vector for additional coupling discretization
    Teuchos::RCP<const Epetra_Vector> coupl_velnp_;

    /// take care of node numbering off-by-one (will be removed soon)
    const bool node_from_zero_;
  };

}  // namespace FLD

BACI_NAMESPACE_CLOSE

#endif  // FLUID_XFLUID_RESULTTEST_H