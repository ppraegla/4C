/*-----------------------------------------------------------*/
/*! \file

\brief time-dependent subgrid scale functionality


\level 3

*/
/*-----------------------------------------------------------*/

#ifndef BACI_FLUID_ELE_TDS_HPP
#define BACI_FLUID_ELE_TDS_HPP

#include "baci_config.hpp"

#include "baci_comm_parobject.hpp"
#include "baci_comm_parobjectfactory.hpp"
#include "baci_linalg_serialdensematrix.hpp"

#include <Teuchos_RCP.hpp>

BACI_NAMESPACE_OPEN


namespace FLD
{
  class TDSEleDataType : public CORE::COMM::ParObjectType
  {
    // friend class ParObjectFactory;
   public:
    static TDSEleDataType& Instance() { return instance_; };

    /// Create ParObject from packed data
    CORE::COMM::ParObject* Create(const std::vector<char>& data) override { return nullptr; }

    /// internal name of this ParObjectType.
    std::string Name() const override { return "TDSEleData"; }

   private:
    static TDSEleDataType instance_;
  };



  class TDSEleData : public CORE::COMM::ParObject
  {
   public:
    /*!
    \brief standard constructor
    */
    TDSEleData();

    /*!
    \brief Pack this class so it can be communicated

    \ref Pack and \ref Unpack are used to communicate this class object

    */
    void Pack(CORE::COMM::PackBuffer& data) const override;

    /*!
    \brief Unpack data from a char vector into this class

    \ref Pack and \ref Unpack are used to communicate this class object
    */
    void Unpack(const std::vector<char>& data) override;

    /*!
    \brief Return unique ParObject id

    every class implementing ParObject needs a unique id defined at the
    top of this file.
    */
    int UniqueParObjectId() const override
    {
      return FLD::TDSEleDataType::Instance().UniqueParObjectId();
    }

    //! @name Time-dependent subgrid scales
    /*!
    \brief Memory allocation for subgrid-scale arrays
    */
    void ActivateTDS(int nquad, int nsd, double** saccn = nullptr, double** sveln = nullptr,
        double** svelnp = nullptr);


    /*!
    \brief Nonlinear update for current subgrid-scale velocities according to the current
           residual (reduced version for afgenalpha and one-step-theta)
    */
    void UpdateSvelnpInOneDirection(const double fac1, const double fac2, const double fac3,
        const double resM, const double alphaF, const int dim, const int iquad, double& svelaf);

    /*!
    \brief Nonlinear update for current subgrid-scale velocities according to the current
           residual (svelnp as additional return value)
    */
    void UpdateSvelnpInOneDirection(const double fac1, const double fac2, const double fac3,
        const double resM, const double alphaF, const int dim, const int iquad, double& svelnp,
        double& svelaf);

    /*!
     * \brief Perform time update of time-dependent subgrid scales
     */
    void Update(const double dt, const double gamma);

    //@}

    /*!
    \brief Returns the subgrid velocity at time n (sveln_)
    */
    CORE::LINALG::SerialDenseMatrix Sveln() const { return sveln_; }

    /*!
    \brief Returns the subgrid velocity at time n+1 (svelnp_)
    */
    CORE::LINALG::SerialDenseMatrix Svelnp() const { return svelnp_; }

   private:
    //! matrices of subgrid-scale acceleration values at integration points of this element
    CORE::LINALG::SerialDenseMatrix saccn_;

    //! matrices of subgrid-scale velocity values, current iteration value, at integration points of
    //! this element
    CORE::LINALG::SerialDenseMatrix svelnp_;

    //! matrices of subgrid-scale velocity values, last timestep, at integration points of this
    //! element
    CORE::LINALG::SerialDenseMatrix sveln_;
  };

}  // namespace FLD

BACI_NAMESPACE_CLOSE

#endif