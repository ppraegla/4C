// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_TIMADA_JOINT_HPP
#define FOUR_C_STRUCTURE_TIMADA_JOINT_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_structure_timada.hpp"
#include "4C_structure_timint.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class Solver;
}

namespace Core::IO
{
  class DiscretizationWriter;
}

/*----------------------------------------------------------------------*/
/* belongs to structure namespace */
namespace Solid
{
  /*====================================================================*/
  /*!
   * \brief Time step size adaptivity with general Solid::TimInt scheme
   *

   */
  template <typename T>
  class TimAdaJoint : public TimAda
  {
   public:
    //! Map Solid::TimInt::NameEnum to Solid::TimAda::NameEnum
    enum Inpar::Solid::TimAdaKind map_name_tim_int_to_tim_ada(
        const enum Inpar::Solid::DynamicType term  //!< input enum term
    ) const
    {
      switch (term)
      {
        case Inpar::Solid::dyna_ab2:
          return Inpar::Solid::timada_kind_ab2;
          break;
        case Inpar::Solid::dyna_expleuler:
          return Inpar::Solid::timada_kind_expleuler;
          break;
        case Inpar::Solid::dyna_centrdiff:
          return Inpar::Solid::timada_kind_centraldiff;
          break;
        default:
          FOUR_C_THROW("Cannot handle requested time integrator");
          return Inpar::Solid::timada_kind_none;
          break;
      }
    }

    //! @name Life
    //@{

    //! Full-fledged constructor
    TimAdaJoint(const Teuchos::ParameterList& ioparams,  //!< ioflags
        const Teuchos::ParameterList& timeparams,        //!< TIS input parameters
        const Teuchos::ParameterList& sdyn,              //!< structural dynamic
        const Teuchos::ParameterList& xparams,           //!< extra flags
        const Teuchos::ParameterList& adaparams,         //!< adaptive input flags
        std::shared_ptr<TimInt>& sti                     //!< marching time integrator
        )
        : TimAda(timeparams, adaparams, sti), ada_(ada_vague), sta_(nullptr)
    {
      // allocate auxiliary integrator
      sta_ = std::make_shared<T>(timeparams, ioparams, sdyn, xparams, sti->discretization(),
          sti->solver(),
          nullptr,  // no contact solver
          sti->disc_writer());
      sta_->init(timeparams, sdyn, xparams, sti->discretization(), sti->solver());

      // check explicitness
      if (sta_->method_implicit())
      {
        FOUR_C_THROW("Implicit might work, but please check carefully");
      }

      // check order
      if (sta_->method_order_of_accuracy_dis() > sti_->method_order_of_accuracy_dis())
      {
        ada_ = ada_upward;
      }
      else if (sta_->method_order_of_accuracy_dis() < sti_->method_order_of_accuracy_dis())
      {
        ada_ = ada_downward;
      }
      else if (sta_->method_name() == sti_->method_name())
      {
        ada_ = ada_ident;
      }
      else
      {
        ada_ = ada_orderequal;
      }

      // setup
      sta_->setup();

      // Actually, we would like to call Merge() and ResizeMStep() now and in the
      // past we have done this right here. However, this requires that both init()
      // and setup() have already been called on both(!) the marching time integrator
      // and the auxiliary time integrator. Since setup() has not yet been called
      // for the marching time integrator, we must postpone Merge() and ResizeMStep()
      // to a later point. They can now be found in the new init() function below,
      // which is called at the beginning of Integrate(). [popp 01/2017]
      // sta_->Merge(*sti);
      // sta_->ResizeMStep();

      return;
    }

    //@}

    //! @name Actions
    //@{

    /*! Finalize the class initialization
     * Merge() and ResizeMStep() need to be called after(!) both init()
     * and setup() have been called on both the marching time integrator
     * and the auxiliary time integrator (popp 01/2017).
     */
    void init(std::shared_ptr<TimInt>& sti) override
    {
      // merge
      sta_->merge(*sti);

      // resize multi-step quantities
      sta_->resize_m_step();

      return;
    }

    /*! \brief Make one step with auxiliary scheme
     *
     *  Afterwards, the auxiliary solutions are stored in the local error
     *  vectors, ie:
     *  - \f$D_{n+1}^{AUX}\f$ in #locdiserrn_
     *  - \f$V_{n+1}^{AUX}\f$ in #locvelerrn_
     */
    void integrate_step_auxiliary() override
    {
      // integrate the auxiliary time integrator one step in time
      sta_->integrate_step();

      // copy onto target
      locerrdisn_->update(1.0, *(sta_->disn_), 0.0);

      // reset
      // remember: sta_ and sti_ are merged and work on the same vectors
      sta_->reset_step();

      return;
    }

    //@}

    //! @name Attributes
    //@{

    //! Provide the name
    enum Inpar::Solid::TimAdaKind method_name() const override
    {
      return map_name_tim_int_to_tim_ada(sti_->method_name());
    }

    //! Provide local order of accuracy of displacements
    int method_order_of_accuracy_dis() const override
    {
      return sta_->method_order_of_accuracy_dis();
    }

    //! Provide local order of accuracy of velocities
    int method_order_of_accuracy_vel() const override
    {
      return sta_->method_order_of_accuracy_vel();
    }

    //! Return linear error coefficient of displacements
    double method_lin_err_coeff_dis() const override { return sta_->method_lin_err_coeff_dis(); }

    //! Return linear error coefficient of velocities
    double method_lin_err_coeff_vel() const override { return sta_->method_lin_err_coeff_vel(); }

    //! Provide type of algorithm
    enum AdaEnum method_adapt_dis() const override { return ada_; }

    //@}

   protected:
    //! not wanted: = operator
    TimAdaJoint operator=(const TimAdaJoint& old);

    //! not wanted: copy constructor
    TimAdaJoint(const TimAdaJoint& old);

    //! type of adaptivity algorithm
    enum AdaEnum ada_;

    //! The auxiliary integrator
    std::shared_ptr<T> sta_;
  };

}  // namespace Solid



/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
