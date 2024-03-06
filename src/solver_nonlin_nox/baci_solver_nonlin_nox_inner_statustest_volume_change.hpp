/*-----------------------------------------------------------*/
/*! \file



\level 3

*/
/*-----------------------------------------------------------*/

#ifndef BACI_SOLVER_NONLIN_NOX_INNER_STATUSTEST_VOLUME_CHANGE_HPP
#define BACI_SOLVER_NONLIN_NOX_INNER_STATUSTEST_VOLUME_CHANGE_HPP

#include "baci_config.hpp"

#include "baci_solver_nonlin_nox_forward_decl.hpp"
#include "baci_solver_nonlin_nox_inner_statustest_generic.hpp"

#include <NOX_Abstract_Group.H>

BACI_NAMESPACE_OPEN

namespace NOX
{
  namespace NLN
  {
    namespace INNER
    {
      namespace StatusTest
      {
        namespace Interface
        {
          class Required;
        }
        /** \brief struct containing the input parameters of the VolumeChange class
         *
         *  \author hiermeier \date 08/18 */
        struct VolumeChangeParams
        {
          /// upper bound ratio
          double upper_bound_ = 1.0;

          /// lower bound ratio
          double lower_bound_ = 1.0;
        };

        /** \brief Test the element volume change
         *
         *  Class which can be used as inner status test to control the admissible
         *  volume change of the related elements from one Newton step to the
         *  next. Therefore, an upper and a lower bound for the volume change ratio
         *  must be provided by the user. If the new element volume is by more than
         *  the upper_bound value larger than the previous element volume, the step
         *  length is rejected. The same is true if the new element volume is by
         *  the lower_bound value smaller than the previously accepted one.
         *
         *  The CheckType should be chosen to \"minimal\" for the status test and
         *  the test should be placed in front of more sophisticated tests such as
         *  the filter test or the Armijo rule, since it can help to detect
         *  cumbersome situation and, thus, avoid the augmentation of inner class
         *  variables with wrong or misleading information.
         *
         *  \author hiermeier \date 08/18 */
        class VolumeChange : public Generic
        {
         public:
          VolumeChange(const VolumeChangeParams& params, const ::NOX::Utils& u)
              : params_(params), utils_(u){};

          StatusType CheckStatus(const Interface::Required& interface,
              const ::NOX::Solver::Generic& solver, const ::NOX::Abstract::Group& grp,
              ::NOX::StatusTest::CheckType checkType) override;

          StatusType GetStatus() const override;

          std::ostream& Print(std::ostream& stream, int indent = 0) const override;

         protected:
          /// set the internal class variables by calling the related evaluate routine
          ::NOX::Abstract::Group::ReturnType SetElementVolumes(
              const ::NOX::Abstract::Group& grp, Teuchos::RCP<Epetra_Vector>& ele_vols) const;

          /// get the number of bad elements
          int NumberOfBadElements();

         protected:
          /// reference element volumes
          Teuchos::RCP<Epetra_Vector> ref_ele_vols_;

          /// current (trial) element volumes
          Teuchos::RCP<Epetra_Vector> curr_ele_vols_;

          /// volume change input-parameters
          const VolumeChangeParams params_;

          /// current number of bad elements
          int num_bad_eles_ = 0;

          /// current number of failing elements (i.e. neg. jacobian determinant)
          int num_failing_eles_ = 0;

          /// maximal value of the current volume ratios
          double max_vol_change_ = 1.0;

          /// minimal value of the current volume ratios
          double min_vol_change_ = 1.0;

          /// global status of the test
          NOX::NLN::INNER::StatusTest::StatusType status_ =
              NOX::NLN::INNER::StatusTest::status_unevaluated;

          /// reference the ostream object
          const ::NOX::Utils& utils_;
        };
      }  // namespace StatusTest
    }    // namespace INNER
  }      // namespace NLN
}  // namespace NOX

BACI_NAMESPACE_CLOSE

#endif  // SOLVER_NONLIN_NOX_INNER_STATUSTEST_VOLUME_CHANGE_H