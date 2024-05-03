/*---------------------------------------------------------------------------*/
/*! \file
\brief Unittests for Parser
\level 3
*/
/*----------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_symbolic_expression.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  /// converts the values of variables from type double to FAD double and returns the modified
  /// vector of name-value-pairs
  std::vector<std::pair<std::string, Sacado::Fad::DFad<double>>> ConvertVariableValuesToFADObjects(
      const std::vector<std::pair<std::string, double>>& variables)
  {
    // prepare return vector
    std::vector<std::pair<std::string, Sacado::Fad::DFad<double>>> variables_FAD;

    // number of variables
    auto numvariables = static_cast<int>(variables.size());

    // counter for variable numbering
    int counter = 0;

    // set the values of the variables
    for (const auto& [name, value] : variables)
    {
      // FAD object for 1st order derivatives
      Sacado::Fad::DFad<double> varfad(numvariables, counter, value);

      // create name-value-pairs with values now of type FAD double and add to vector
      variables_FAD.emplace_back(name, varfad);

      // update counter
      counter++;
    }
    return variables_FAD;
  }


  TEST(SymbolicExpressionTest, TestNoVariables)
  {
    CORE::UTILS::SymbolicExpression<double> symbolicexpression("2.0");

    EXPECT_DOUBLE_EQ(symbolicexpression.Value({}), 2.0);
  }

  TEST(SymbolicExpressionTest, TestValue)
  {
    CORE::UTILS::SymbolicExpression<double> symbolicexpression("2*x");

    EXPECT_DOUBLE_EQ(symbolicexpression.Value({{"x", 2.0}}), 4.0);
  }

  TEST(SymbolicExpressionTest, TestFirstDeriv)
  {
    CORE::UTILS::SymbolicExpression<double> symbolicexpression_bilin(
        "2*Variable1*Constant1*Variable2");
    CORE::UTILS::SymbolicExpression<double> symbolicexpression_xtimesx(
        "2*Variable1*Variable1*Constant1*Variable2*Variable2");
    CORE::UTILS::SymbolicExpression<double> symbolicexpression_pow2(
        "2*Variable1^2*Constant1*Variable2^2");
    std::map<std::string, double> constants{{"Constant1", 2.0}};

    std::vector<std::pair<std::string, double>> variables;
    variables.emplace_back("Variable1", 6.0);
    variables.emplace_back("Variable2", 3.0);

    auto variables_FAD = ConvertVariableValuesToFADObjects(variables);

    // convert vector of pairs to map variables_values
    std::map<std::string, Sacado::Fad::DFad<double>> variable_values;

    std::copy(variables_FAD.begin(), variables_FAD.end(),
        std::inserter(variable_values, variable_values.begin()));

    auto fdfad_bilin = symbolicexpression_bilin.FirstDerivative(variable_values, constants);
    auto fdfad_xtimesx = symbolicexpression_xtimesx.FirstDerivative(variable_values, constants);
    auto fdfad_pow2 = symbolicexpression_pow2.FirstDerivative(variable_values, constants);

    EXPECT_DOUBLE_EQ(fdfad_bilin.dx(0), 12.0);                    // dFunction1/dVariable1
    EXPECT_DOUBLE_EQ(fdfad_bilin.dx(1), 24.0);                    // dFunction1/dVariable2
    EXPECT_NEAR(fdfad_xtimesx.dx(0), fdfad_pow2.dx(0), 1.0e-14);  // dFunction2/dVariable1
    EXPECT_NEAR(fdfad_xtimesx.dx(1), fdfad_pow2.dx(1), 1.0e-14);  // dFunction2/dVariable1
  }

  TEST(SymbolicExpressionTest, TestValidFunctionsAndOperators)
  {
    CORE::UTILS::SymbolicExpression<double> symbolicexpression_sincostan(
        "2*cos(x) * sin(x) * tan(x) + cosh(x) * sinh(x) * tanh(x) + asin(1.0) * acos(0.5) * "
        "atan(1.0) ");

    CORE::UTILS::SymbolicExpression<double> symbolicexpression_logexp(
        " log(exp(1)) * log10(y) - x");

    CORE::UTILS::SymbolicExpression<double> symbolicexpression_sqrtheavisidefabs(
        "sqrt(4) + heaviside(3.0) + fabs(2.3) / 1^1");

    CORE::UTILS::SymbolicExpression<double> symbolicexpression_atan2("atan2(2,4)");

    CORE::UTILS::SymbolicExpression<double> symbolicexpression_xpow2("x^2");
    CORE::UTILS::SymbolicExpression<double> symbolicexpression_xtimesx("x * x");

    EXPECT_NEAR(
        symbolicexpression_sincostan.Value({{"x", 0.2}, {"y", 0.4}}), 1.4114033869288349, 1.0e-14);

    EXPECT_NEAR(
        symbolicexpression_logexp.Value({{"x", 0.2}, {"y", 0.4}}), -0.59794000867203767, 1.0e-14);

    EXPECT_NEAR(symbolicexpression_sqrtheavisidefabs.Value({}), 5.3, 1.0e-14);

    EXPECT_NEAR(symbolicexpression_atan2.Value({}), 0.46364760900080609, 1.0e-14);
    EXPECT_NEAR(symbolicexpression_xpow2.Value({{"x", 0.2}}),
        symbolicexpression_xtimesx.Value({{"x", 0.2}}), 1.0e-14);
  }

  TEST(SymbolicExpressionTest, TestValidLiterals)
  {
    CORE::UTILS::SymbolicExpression<double> symbolicexpression("2*pi * 1.0e-3  + 3.0E-4 * x");

    EXPECT_NEAR(symbolicexpression.Value({{"x", 1.0}}), 0.0065831853071795865, 1.0e-14);
  }

  TEST(SymbolicExpressionTest, EvaluateWithMissingVariableThrows)
  {
    CORE::UTILS::SymbolicExpression<double> symbolicexpression(
        "2*Variable1*Constant1*Variable2*Variable3");

    EXPECT_ANY_THROW(symbolicexpression.Value({{"Variable1", 1.0}, {"Constant1", 1.0}}));
  }

  TEST(SymbolicExpressionTest, InvalidOperatorThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        CORE::UTILS::SymbolicExpression<double> symbolicexpression("2 ** 4"), CORE::Exception,
        "unexpected token 7");
  }


  TEST(SymbolicExpressionTest, MissingBracketsThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        CORE::UTILS::SymbolicExpression<double> symbolicexpression("2*4 - (3 + 1"), CORE::Exception,
        "')' expected");
  }

  TEST(SymbolicExpressionTest, IncompleteFunctionThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        CORE::UTILS::SymbolicExpression<double> symbolicexpression("2*4 - (3 + "), CORE::Exception,
        "unexpected token 1");
  }



}  // namespace
FOUR_C_NAMESPACE_CLOSE