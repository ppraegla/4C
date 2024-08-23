/*! \file

    * \brief Declaration of the solid-poro pressure-velocity-based element
    * This file contains the element-specific service routines such as Pack, Unpack, NumDofPerNode
etc.
\level 1
                */

#ifndef FOUR_C_SOLID_PORO_3D_ELE_PRESSURE_VELOCITY_BASED_HPP
#define FOUR_C_SOLID_PORO_3D_ELE_PRESSURE_VELOCITY_BASED_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_inpar_poro.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_solid_3D_ele_factory.hpp"
#include "4C_solid_poro_3D_ele_factory.hpp"
#include "4C_solid_poro_3D_ele_properties.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  class StructPoro;
  class FluidPoro;
}  // namespace Mat

namespace Solid::ELEMENTS
{
  enum class EasType;
}

namespace Discret::ELEMENTS
{

  class SolidPoroPressureVelocityBasedType : public Core::Elements::ElementType
  {
   public:
    void setup_element_definition(
        std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions) override;

    Teuchos::RCP<Core::Elements::Element> create(const std::string eletype,
        const std::string elecelltype, const int id, const int owner) override;

    Teuchos::RCP<Core::Elements::Element> create(const int id, const int owner) override;

    Core::Communication::ParObject* create(const std::vector<char>& data) override;

    [[nodiscard]] std::string name() const override { return "SolidPoroPressureVelocityBasedType"; }

    void nodal_block_information(
        Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

    static SolidPoroPressureVelocityBasedType& instance();

   private:
    static SolidPoroPressureVelocityBasedType instance_;

  };  // class SolidPoroType


  class SolidPoroPressureVelocityBased : public Core::Elements::Element
  {
    friend class SolidPoroType;

   public:
    //! @name Constructors and destructors and related methods
    //!@{

    /*!
    \brief Standard Constructor

    \param id    (in): A globally unique element id
    \param owner (in): owner processor of the element
    */
    SolidPoroPressureVelocityBased(int id, int owner);

    //!@}

    [[nodiscard]] Core::Elements::Element* clone() const override;

    [[nodiscard]] int unique_par_object_id() const override
    {
      return SolidPoroPressureVelocityBasedType::instance().unique_par_object_id();
    };

    [[nodiscard]] int num_line() const override;

    [[nodiscard]] int num_surface() const override;

    [[nodiscard]] int num_volume() const override;

    std::vector<Teuchos::RCP<Core::Elements::Element>> lines() override;

    std::vector<Teuchos::RCP<Core::Elements::Element>> surfaces() override;

    [[nodiscard]] int num_dof_per_node(const Core::Nodes::Node& node) const override { return 3; }

    [[nodiscard]] int num_dof_per_element() const override { return 0; }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(const std::vector<char>& data) override;

    [[nodiscard]] Core::FE::CellType shape() const override { return celltype_; };

    [[nodiscard]] Core::Elements::ElementType& element_type() const override
    {
      return SolidPoroPressureVelocityBasedType::instance();
    }

    bool read_element(const std::string& eletype, const std::string& celltype,
        const Core::IO::InputParameterContainer& container) override;

    int evaluate(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        Core::Elements::Element::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
        Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseVector& elevec2,
        Core::LinAlg::SerialDenseVector& elevec3) override;

    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

    void read_anisotropic_permeability_directions_from_element_line_definition(
        const Core::IO::InputParameterContainer& container);

    void read_anisotropic_permeability_nodal_coeffs_from_element_line_definition(
        const Core::IO::InputParameterContainer& container);

    void set_params_interface_ptr(const Teuchos::ParameterList& p) override;

    Teuchos::RCP<Core::Elements::ParamsInterface> params_interface_ptr() override
    {
      return interface_ptr_;
    }

    [[nodiscard]] inline bool is_solid_params_interface() const
    {
      return (not solid_interface_ptr_.is_null());
    }

    [[nodiscard]] inline bool is_params_interface() const override
    {
      return (not interface_ptr_.is_null());
    }

    [[nodiscard]] inline Core::Elements::ParamsInterface& params_interface() const
    {
      if (not is_params_interface()) FOUR_C_THROW("The interface ptr is not set!");
      return *interface_ptr_;
    }

    [[nodiscard]] inline FourC::Solid::ELEMENTS::ParamsInterface& get_solid_params_interface() const
    {
      FOUR_C_THROW_UNLESS(solid_interface_ptr_.getRawPtr(),
          "The parameter interface pointer is not set or not a solid parameter interface.");
      return *solid_interface_ptr_;
    }

    [[nodiscard]] Mat::StructPoro& struct_poro_material(int nummat = 0) const;

    [[nodiscard]] Mat::FluidPoro& fluid_poro_material(int nummat = 1) const;

    [[nodiscard]] Mat::So3Material& solid_poro_material(int nummat = 0) const;


    Inpar::Solid::KinemType kinematic_type() { return solid_ele_property_.kintype; }

    [[nodiscard]] const AnisotropyProperties& get_anisotropic_permeability_property() const
    {
      return anisotropic_permeability_property_;
    }

    //! return anisotropic permeability directions (used for cloning)
    [[nodiscard]] const std::vector<std::vector<double>>& get_anisotropic_permeability_directions()
        const
    {
      return anisotropic_permeability_property_.directions_;
    }

    //! return scaling coefficients for anisotropic permeability (used for cloning)
    [[nodiscard]] const std::vector<std::vector<double>>&
    get_anisotropic_permeability_nodal_coeffs() const
    {
      return anisotropic_permeability_property_.nodal_coeffs_;
    }

    void vis_names(std::map<std::string, int>& names) override;

    bool vis_data(const std::string& name, std::vector<double>& data) override;

   private:
    //! cell type
    Core::FE::CellType celltype_{Core::FE::CellType::dis_none};

    //! solid element properties
    SolidElementProperties solid_ele_property_{};

    //! additional poro element properties
    SolidPoroElementProperties poro_ele_property_{};

    //! anisotropy element properties
    AnisotropyProperties anisotropic_permeability_property_{};

    //! interface pointer for data exchange between the element and the time integrator.
    Teuchos::RCP<Core::Elements::ParamsInterface> interface_ptr_;

    //! interface pointer for data exchange between the element and the solid time integrator.
    Teuchos::RCP<FourC::Solid::ELEMENTS::ParamsInterface> solid_interface_ptr_;

    //! element calculation holding one of the implemented variants
    SolidCalcVariant solid_calc_variant_;

    //! poro element calculation holding one of the implemented variants
    SolidPoroPressureVelocityBasedCalcVariant solidporo_press_vel_based_calc_variant_;

    //! flag, whether the post setup of materials is already called
    bool material_post_setup_ = false;

  };  // class SolidPoroPressureVelocityBased

}  // namespace Discret::ELEMENTS

FOUR_C_NAMESPACE_CLOSE

#endif