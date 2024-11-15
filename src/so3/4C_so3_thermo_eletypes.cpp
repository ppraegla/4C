// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_thermo_eletypes.hpp"

#include "4C_io_linedefinition.hpp"
#include "4C_so3_thermo.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *  HEX8 element
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | build an instance of thermo type                          dano 08/12 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex8ThermoType Discret::Elements::SoHex8ThermoType::instance_;


/*----------------------------------------------------------------------*
 | access an instance of thermo type                                    |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex8ThermoType& Discret::Elements::SoHex8ThermoType::instance()
{
  return instance_;
}


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 08/12 |
 | is called in ElementRegisterType                                     |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::SoHex8ThermoType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3Thermo<Discret::Elements::SoHex8, Core::FE::CellType::hex8>* object =
      new Discret::Elements::So3Thermo<Discret::Elements::SoHex8, Core::FE::CellType::hex8>(-1, -1);
  object->unpack(buffer);
  return object;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 08/12 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoHex8ThermoType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SOLIDH8THERMO")
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::Elements::So3Thermo<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 08/12 |
 | virtual method of ElementType                                        |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoHex8ThermoType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::Elements::So3Thermo<Discret::Elements::SoHex8, Core::FE::CellType::hex8>>(

      id, owner);
  return ele;

}  // Create()


/*----------------------------------------------------------------------*
 | setup the element definition (public)                     dano 08/12 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8ThermoType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_hex8;
  SoHex8Type::setup_element_definition(definitions_hex8);

  std::map<std::string, Input::LineDefinition>& defs_hex8 = definitions_hex8["SOLIDH8_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["HEX8"] = defs_hex8["HEX8"];

}  // setup_element_definition()


/*----------------------------------------------------------------------*
 | initialise the element (public)                           dano 08/12 |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex8ThermoType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;

    Discret::Elements::So3Thermo<Discret::Elements::SoHex8, Core::FE::CellType::hex8>* actele =
        dynamic_cast<
            Discret::Elements::So3Thermo<Discret::Elements::SoHex8, Core::FE::CellType::hex8>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex8_thermo* failed");
    // initialise all quantities
    actele->SoHex8::init_jacobian_mapping();
    // as an alternative we can call: So_hex8Type::initialize(dis);
    actele->So3Thermo<Discret::Elements::SoHex8,
        Core::FE::CellType::hex8>::init_jacobian_mapping_special_for_nurbs(dis);
  }

  return 0;
}  // initialize()
/*----------------------------------------------------------------------------*
 | ENDE HEX8 Element
 *----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*
 *  HEX8FBAR element
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | build an instance of thermo type                          dano 05/13 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex8fbarThermoType Discret::Elements::SoHex8fbarThermoType::instance_;

Discret::Elements::SoHex8fbarThermoType& Discret::Elements::SoHex8fbarThermoType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 05/13 |
 | is called in ElementRegisterType                                     |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::SoHex8fbarThermoType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3Thermo<Discret::Elements::SoHex8fbar, Core::FE::CellType::hex8>* object =
      new Discret::Elements::So3Thermo<Discret::Elements::SoHex8fbar, Core::FE::CellType::hex8>(
          -1, -1);
  object->unpack(buffer);
  return object;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 05/13 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoHex8fbarThermoType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SOLIDH8FBARTHERMO")
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::Elements::So3Thermo<Discret::Elements::SoHex8fbar, Core::FE::CellType::hex8>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 05/13 |
 | virtual method of ElementType                                        |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoHex8fbarThermoType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::Elements::So3Thermo<Discret::Elements::SoHex8fbar, Core::FE::CellType::hex8>>(

      id, owner);
  return ele;
}  // Create()


/*----------------------------------------------------------------------*
 | setup the element definition (public)                     dano 05/13 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8fbarThermoType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  // original definition
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_hex8fbar;

  // call setup of So3Ele
  SoHex8fbarType::setup_element_definition(definitions_hex8fbar);

  std::map<std::string, Input::LineDefinition>& defs_hex8fbar =
      definitions_hex8fbar["SOLIDH8FBAR_DEPRECATED"];

  // templated definition
  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["HEX8"] = defs_hex8fbar["HEX8"];

}  // setup_element_definition()


/*----------------------------------------------------------------------*
 | initialise the element (public)                           dano 05/13 |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex8fbarThermoType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;

    Discret::Elements::So3Thermo<Discret::Elements::SoHex8fbar, Core::FE::CellType::hex8>* actele =
        dynamic_cast<
            Discret::Elements::So3Thermo<Discret::Elements::SoHex8fbar, Core::FE::CellType::hex8>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex8fbar_thermo* failed");

    // initialise all quantities
    actele->SoHex8fbar::init_jacobian_mapping();
    // as an alternative we can call: So_hex8fbarType::initialize(dis);
    actele->So3Thermo<Discret::Elements::SoHex8fbar,
        Core::FE::CellType::hex8>::init_jacobian_mapping_special_for_nurbs(dis);
  }

  return 0;
}  // initialize()
/*----------------------------------------------------------------------------*
 | ENDE HEX8FBAR Element
 *----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*
 *  TET4 element
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | build an instance of thermo type                          dano 08/12 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoTet4ThermoType Discret::Elements::SoTet4ThermoType::instance_;

Discret::Elements::SoTet4ThermoType& Discret::Elements::SoTet4ThermoType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 08/12 |
 | is called in ElementRegisterType                                     |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::SoTet4ThermoType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3Thermo<Discret::Elements::SoTet4, Core::FE::CellType::tet4>* object =
      new Discret::Elements::So3Thermo<Discret::Elements::SoTet4, Core::FE::CellType::tet4>(-1, -1);
  object->unpack(buffer);
  return object;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 08/12 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoTet4ThermoType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SOLIDT4THERMO")
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::Elements::So3Thermo<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 08/12 |
 | virtual method of ElementType                                        |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoTet4ThermoType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::Elements::So3Thermo<Discret::Elements::SoTet4, Core::FE::CellType::tet4>>(

      id, owner);
  return ele;
}  // Create()


/*----------------------------------------------------------------------*
 | build an instance of thermo type                          dano 08/12 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoTet4ThermoType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_tet4;
  SoTet4Type::setup_element_definition(definitions_tet4);

  std::map<std::string, Input::LineDefinition>& defs_tet4 = definitions_tet4["SOLIDT4_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["TET4"] = defs_tet4["TET4"];

}  // setup_element_definition()


/*----------------------------------------------------------------------*
 | initialise the element (public)                           dano 08/12 |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoTet4ThermoType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;

    Discret::Elements::So3Thermo<Discret::Elements::SoTet4, Core::FE::CellType::tet4>* actele =
        dynamic_cast<
            Discret::Elements::So3Thermo<Discret::Elements::SoTet4, Core::FE::CellType::tet4>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_tet4_thermo* failed");

    actele->SoTet4::init_jacobian_mapping();
    // as an alternative we can call: So_tet4Type::initialize(dis);
    actele->So3Thermo<Discret::Elements::SoTet4,
        Core::FE::CellType::tet4>::init_jacobian_mapping_special_for_nurbs(dis);
  }

  return 0;
}  // initialize()
/*----------------------------------------------------------------------------*
 | ENDE TET4 Element
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*
 *  TET10 element
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | build an instance of thermo type                         farah 05/14 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoTet10ThermoType Discret::Elements::SoTet10ThermoType::instance_;

Discret::Elements::SoTet10ThermoType& Discret::Elements::SoTet10ThermoType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 | create the new element type (public)                     farah 05/14 |
 | is called in ElementRegisterType                                     |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::SoTet10ThermoType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3Thermo<Discret::Elements::SoTet10, Core::FE::CellType::tet10>* object =
      new Discret::Elements::So3Thermo<Discret::Elements::SoTet10, Core::FE::CellType::tet10>(
          -1, -1);
  object->unpack(buffer);
  return object;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                     farah 05/14 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoTet10ThermoType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SOLIDT10THERMO")
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::Elements::So3Thermo<Discret::Elements::SoTet10, Core::FE::CellType::tet10>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                     farah 05/14 |
 | virtual method of ElementType                                        |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoTet10ThermoType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::Elements::So3Thermo<Discret::Elements::SoTet10, Core::FE::CellType::tet10>>(

      id, owner);
  return ele;
}  // Create()


/*----------------------------------------------------------------------*
 | build an instance of thermo type                         farah 05/14 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoTet10ThermoType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_tet10;
  SoTet10Type::setup_element_definition(definitions_tet10);

  std::map<std::string, Input::LineDefinition>& defs_tet10 =
      definitions_tet10["SOLIDT10_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["TET10"] = defs_tet10["TET10"];

}  // setup_element_definition()


/*----------------------------------------------------------------------*
 | initialise the element (public)                          farah 05/14 |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoTet10ThermoType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;

    Discret::Elements::So3Thermo<Discret::Elements::SoTet10, Core::FE::CellType::tet10>* actele =
        dynamic_cast<
            Discret::Elements::So3Thermo<Discret::Elements::SoTet10, Core::FE::CellType::tet10>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_tet10_thermo* failed");

    actele->SoTet10::init_jacobian_mapping();
    // as an alternative we can call: So_tet4Type::initialize(dis);
    actele->So3Thermo<Discret::Elements::SoTet10,
        Core::FE::CellType::tet10>::init_jacobian_mapping_special_for_nurbs(dis);
  }

  return 0;
}  // initialize()
/*----------------------------------------------------------------------------*
 | ENDE TET10 Element
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  HEX 27 Element
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | build an instance of thermo type                          dano 10/13 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex27ThermoType Discret::Elements::SoHex27ThermoType::instance_;

Discret::Elements::SoHex27ThermoType& Discret::Elements::SoHex27ThermoType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 10/13 |
 | is called in ElementRegisterType                                     |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::SoHex27ThermoType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3Thermo<Discret::Elements::SoHex27, Core::FE::CellType::hex27>* object =
      new Discret::Elements::So3Thermo<Discret::Elements::SoHex27, Core::FE::CellType::hex27>(
          -1, -1);
  object->unpack(buffer);
  return object;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 10/13 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoHex27ThermoType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SOLIDH27THERMO")
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::Elements::So3Thermo<Discret::Elements::SoHex27, Core::FE::CellType::hex27>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                      dano 10/13 |
 | virtual method of ElementType                                        |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoHex27ThermoType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::Elements::So3Thermo<Discret::Elements::SoHex27, Core::FE::CellType::hex27>>(

      id, owner);
  return ele;
}  // Create ()


/*----------------------------------------------------------------------*
 | setup the element definition (public)                     dano 10/13 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex27ThermoType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_hex27;
  SoHex27Type::setup_element_definition(definitions_hex27);

  std::map<std::string, Input::LineDefinition>& defs_hex27 =
      definitions_hex27["SOLIDH27_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["HEX27"] = defs_hex27["HEX27"];

}  // setup_element_definition()


/*----------------------------------------------------------------------*
 | initialise the element (public)                           dano 10/13 |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex27ThermoType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;

    Discret::Elements::So3Thermo<Discret::Elements::SoHex27, Core::FE::CellType::hex27>* actele =
        dynamic_cast<
            Discret::Elements::So3Thermo<Discret::Elements::SoHex27, Core::FE::CellType::hex27>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex27_thermo* failed");

    actele->SoHex27::init_jacobian_mapping();
    // as an alternative we can call: So_hex27Type::initialize(dis);
    actele->So3Thermo<Discret::Elements::SoHex27,
        Core::FE::CellType::hex27>::init_jacobian_mapping_special_for_nurbs(dis);
  }

  return 0;
}  // initialize()
/*----------------------------------------------------------------------------*
 | ENDE HEX27 Element
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  HEX 20 Element
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | build an instance of thermo type                         farah 05/14 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoHex20ThermoType Discret::Elements::SoHex20ThermoType::instance_;

Discret::Elements::SoHex20ThermoType& Discret::Elements::SoHex20ThermoType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 | create the new element type (public)                     farah 05/14 |
 | is called in ElementRegisterType                                     |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::SoHex20ThermoType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3Thermo<Discret::Elements::SoHex20, Core::FE::CellType::hex20>* object =
      new Discret::Elements::So3Thermo<Discret::Elements::SoHex20, Core::FE::CellType::hex20>(
          -1, -1);
  object->unpack(buffer);
  return object;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                     farah 05/14 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoHex20ThermoType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SOLIDH20THERMO")
  {
    Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
        Discret::Elements::So3Thermo<Discret::Elements::SoHex20, Core::FE::CellType::hex20>>(

        id, owner);
    return ele;
  }
  return Teuchos::null;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                     farah 05/14 |
 | virtual method of ElementType                                        |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoHex20ThermoType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele = Teuchos::make_rcp<
      Discret::Elements::So3Thermo<Discret::Elements::SoHex20, Core::FE::CellType::hex20>>(

      id, owner);
  return ele;
}  // Create ()


/*----------------------------------------------------------------------*
 | setup the element definition (public)                    farah 05/14 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex20ThermoType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_hex20;
  SoHex20Type::setup_element_definition(definitions_hex20);

  std::map<std::string, Input::LineDefinition>& defs_hex20 =
      definitions_hex20["SOLIDH20_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["HEX20"] = defs_hex20["HEX20"];

}  // setup_element_definition()


/*----------------------------------------------------------------------*
 | initialise the element (public)                          farah 05/14 |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex20ThermoType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;

    Discret::Elements::So3Thermo<Discret::Elements::SoHex20, Core::FE::CellType::hex20>* actele =
        dynamic_cast<
            Discret::Elements::So3Thermo<Discret::Elements::SoHex20, Core::FE::CellType::hex20>*>(
            dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex20_thermo* failed");

    actele->SoHex20::init_jacobian_mapping();
    // as an alternative we can call: So_hex27Type::initialize(dis);
    actele->So3Thermo<Discret::Elements::SoHex20,
        Core::FE::CellType::hex20>::init_jacobian_mapping_special_for_nurbs(dis);
  }

  return 0;
}  // initialize()
/*----------------------------------------------------------------------------*
 | ENDE HEX20 Element
 *----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 |  nurbs 27 Element
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | build an instance of thermo type                         seitz 12/15 |
 *----------------------------------------------------------------------*/
Discret::Elements::SoNurbs27ThermoType Discret::Elements::SoNurbs27ThermoType::instance_;

Discret::Elements::SoNurbs27ThermoType& Discret::Elements::SoNurbs27ThermoType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 | create the new element type (public)                     seitz 12/15 |
 | is called in ElementRegisterType                                     |
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::SoNurbs27ThermoType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::So3Thermo<Discret::Elements::Nurbs::SoNurbs27, Core::FE::CellType::nurbs27>*
      object = new Discret::Elements::So3Thermo<Discret::Elements::Nurbs::SoNurbs27,
          Core::FE::CellType::nurbs27>(-1, -1);
  object->unpack(buffer);
  return object;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                     seitz 12/15 |
 | is called from ParObjectFactory                                      |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoNurbs27ThermoType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SONURBS27THERMO")
  {
    Teuchos::RCP<Core::Elements::Element> ele =
        Teuchos::make_rcp<Discret::Elements::So3Thermo<Discret::Elements::Nurbs::SoNurbs27,
            Core::FE::CellType::nurbs27>>(id, owner);
    return ele;
  }
  return Teuchos::null;
}  // Create()


/*----------------------------------------------------------------------*
 | create the new element type (public)                     seitz 12/15 |
 | virtual method of ElementType                                        |
 *----------------------------------------------------------------------*/
Teuchos::RCP<Core::Elements::Element> Discret::Elements::SoNurbs27ThermoType::create(
    const int id, const int owner)
{
  Teuchos::RCP<Core::Elements::Element> ele =
      Teuchos::make_rcp<Discret::Elements::So3Thermo<Discret::Elements::Nurbs::SoNurbs27,
          Core::FE::CellType::nurbs27>>(id, owner);
  return ele;
}  // Create ()


/*----------------------------------------------------------------------*
 | setup the element definition (public)                    seitz 12/15 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoNurbs27ThermoType::setup_element_definition(
    std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
{
  std::map<std::string, std::map<std::string, Input::LineDefinition>> definitions_nurbs27;
  Nurbs::SoNurbs27Type::setup_element_definition(definitions_nurbs27);

  std::map<std::string, Input::LineDefinition>& defs_nurbs27 =
      definitions_nurbs27["SONURBS27_DEPRECATED"];

  std::map<std::string, Input::LineDefinition>& defs = definitions[get_element_type_string()];

  defs["NURBS27"] = defs_nurbs27["NURBS27"];

}  // setup_element_definition()


/*----------------------------------------------------------------------*
 | initialise the element (public)                          seitz 12/15 |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoNurbs27ThermoType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;

    Discret::Elements::So3Thermo<Discret::Elements::Nurbs::SoNurbs27, Core::FE::CellType::nurbs27>*
        actele = dynamic_cast<Discret::Elements::So3Thermo<Discret::Elements::Nurbs::SoNurbs27,
            Core::FE::CellType::nurbs27>*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex20_thermo* failed");

    actele->SoNurbs27::init_jacobian_mapping(dis);
    // as an alternative we can call: So_hex27Type::initialize(dis);
    actele->So3Thermo<Discret::Elements::Nurbs::SoNurbs27,
        Core::FE::CellType::nurbs27>::init_jacobian_mapping_special_for_nurbs(dis);
  }

  return 0;
}  // initialize()
/*----------------------------------------------------------------------------*
 | END nurbs27 Element
 *----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
