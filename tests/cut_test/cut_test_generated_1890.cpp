// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_combintersection.hpp"
#include "4C_cut_levelsetintersection.hpp"
#include "4C_cut_meshintersection.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_tetmeshintersection.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cut_test_utils.hpp"

void test_generated_1890()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2542);
    tri3_xyze(0, 1) = -0.0595492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2546);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-1);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0595492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2546);
    tri3_xyze(0, 1) = -0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-1);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2582);
    tri3_xyze(0, 1) = -0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2542);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-2);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2542);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-2);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2581);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-2);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2577);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-3);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 1) = -0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-3);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 1) = -0.0595492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2579);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-3);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 1) = -0.0595492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2546);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-7);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0595492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2546);
    tri3_xyze(0, 1) = -0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2802);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-7);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2802);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2801);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-7);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2801);
    tri3_xyze(0, 1) = -0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-7);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0595492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2579);
    tri3_xyze(0, 1) = -0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-8);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2801);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-8);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2801);
    tri3_xyze(0, 1) = -0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2819);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-8);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0595492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2546);
    tri3_xyze(0, 1) = -0.0845492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2551);
    tri3_xyze(0, 2) = -0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-9);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0845492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2551);
    tri3_xyze(0, 1) = -0.0816682;
    tri3_xyze(1, 1) = 0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2805);
    tri3_xyze(0, 2) = -0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-9);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2802);
    tri3_xyze(0, 1) = -0.0595492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2546);
    tri3_xyze(0, 2) = -0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-9);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 1) = -0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2542);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-1);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2549);
    tri3_xyze(0, 1) = -0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2553);
    tri3_xyze(0, 2) = -0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-10);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2553);
    tri3_xyze(0, 1) = -0.115451;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2555);
    tri3_xyze(0, 2) = -0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-10);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.115451;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2555);
    tri3_xyze(0, 1) = -0.0845492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2551);
    tri3_xyze(0, 2) = -0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-10);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0845492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2551);
    tri3_xyze(0, 1) = -0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2549);
    tri3_xyze(0, 2) = -0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-10);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2585);
    tri3_xyze(0, 1) = -0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2587);
    tri3_xyze(0, 2) = -0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-11);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2587);
    tri3_xyze(0, 1) = -0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2553);
    tri3_xyze(0, 2) = -0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-11);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2553);
    tri3_xyze(0, 1) = -0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2549);
    tri3_xyze(0, 2) = -0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-11);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2549);
    tri3_xyze(0, 1) = -0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2585);
    tri3_xyze(0, 2) = -0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-11);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0845492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2551);
    tri3_xyze(0, 1) = -0.115451;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2555);
    tri3_xyze(0, 2) = -0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-12);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0816682;
    tri3_xyze(1, 0) = 0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2805);
    tri3_xyze(0, 1) = -0.0845492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2551);
    tri3_xyze(0, 2) = -0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-12);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2545);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 2) = -0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-1);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2581);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2582);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-2);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2599);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2581);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-4);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2581);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-4);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2541);
    tri3_xyze(0, 1) = -0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2577);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-4);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2542);
    tri3_xyze(0, 1) = -0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2549);
    tri3_xyze(0, 2) = -0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-5);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2549);
    tri3_xyze(0, 1) = -0.0845492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2551);
    tri3_xyze(0, 2) = -0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-5);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0845492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2551);
    tri3_xyze(0, 1) = -0.0595492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2546);
    tri3_xyze(0, 2) = -0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-5);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0595492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2546);
    tri3_xyze(0, 1) = -0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2542);
    tri3_xyze(0, 2) = -0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-5);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2582);
    tri3_xyze(0, 1) = -0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2585);
    tri3_xyze(0, 2) = -0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-6);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2585);
    tri3_xyze(0, 1) = -0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2549);
    tri3_xyze(0, 2) = -0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-6);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2549);
    tri3_xyze(0, 1) = -0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2542);
    tri3_xyze(0, 2) = -0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-6);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2542);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2582);
    tri3_xyze(0, 2) = -0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-6);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2601);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2602);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-31);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2602);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2582);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-31);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2582);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2581);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-31);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2581);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2601);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-31);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2619);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2601);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-32);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2601);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2581);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-32);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2581);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2599);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-32);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2599);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2619);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-32);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2602);
    tri3_xyze(0, 1) = -0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2605);
    tri3_xyze(0, 2) = -0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-33);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2605);
    tri3_xyze(0, 1) = -0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2585);
    tri3_xyze(0, 2) = -0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-33);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2585);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2582);
    tri3_xyze(0, 2) = -0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-33);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2582);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2602);
    tri3_xyze(0, 2) = -0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-33);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2587);
    tri3_xyze(0, 1) = -0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2585);
    tri3_xyze(0, 2) = -0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-34);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2585);
    tri3_xyze(0, 1) = -0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2605);
    tri3_xyze(0, 2) = -0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-34);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2617);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2619);
    tri3_xyze(0, 2) = -0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-40);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2619);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2599);
    tri3_xyze(0, 2) = -0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-40);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2621);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2622);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-41);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2622);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2602);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-41);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2602);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2601);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-41);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2601);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2621);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-41);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2639);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2621);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-42);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2621);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2601);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-42);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2601);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2619);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-42);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2619);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2639);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-42);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2622);
    tri3_xyze(0, 1) = -0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2625);
    tri3_xyze(0, 2) = -0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-43);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2625);
    tri3_xyze(0, 1) = -0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2605);
    tri3_xyze(0, 2) = -0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-43);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2605);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2602);
    tri3_xyze(0, 2) = -0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-43);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2602);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2622);
    tri3_xyze(0, 2) = -0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-43);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2625);
    tri3_xyze(0, 1) = -0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2627);
    tri3_xyze(0, 2) = -0.0603553;
    tri3_xyze(1, 2) = -0.0786566;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-44);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2605);
    tri3_xyze(0, 1) = -0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2625);
    tri3_xyze(0, 2) = -0.0603553;
    tri3_xyze(1, 2) = -0.0786566;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-44);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2639);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2619);
    tri3_xyze(0, 2) = -0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-50);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2619);
    tri3_xyze(0, 1) = -0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2617);
    tri3_xyze(0, 2) = -0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-50);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2641);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2642);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-51);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2642);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2622);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-51);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2622);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2621);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-51);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2621);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2641);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-51);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2659);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2641);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-52);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2641);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2621);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-52);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2621);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2639);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-52);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2642);
    tri3_xyze(0, 1) = -0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2645);
    tri3_xyze(0, 2) = -0.0273361;
    tri3_xyze(1, 2) = -0.0659953;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-53);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2645);
    tri3_xyze(0, 1) = -0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2625);
    tri3_xyze(0, 2) = -0.0273361;
    tri3_xyze(1, 2) = -0.0659953;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-53);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2625);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2622);
    tri3_xyze(0, 2) = -0.0273361;
    tri3_xyze(1, 2) = -0.0659953;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-53);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2622);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2642);
    tri3_xyze(0, 2) = -0.0273361;
    tri3_xyze(1, 2) = -0.0659953;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-53);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2645);
    tri3_xyze(0, 1) = -0.0298809;
    tri3_xyze(1, 1) = -0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2647);
    tri3_xyze(0, 2) = -0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-54);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0298809;
    tri3_xyze(1, 0) = -0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2647);
    tri3_xyze(0, 1) = -0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2627);
    tri3_xyze(0, 2) = -0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-54);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0577254;
    tri3_xyze(1, 0) = -0.0999834;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2627);
    tri3_xyze(0, 1) = -0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2625);
    tri3_xyze(0, 2) = -0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-54);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2625);
    tri3_xyze(0, 1) = -0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2645);
    tri3_xyze(0, 2) = -0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-54);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01416e-15;
    tri3_xyze(1, 0) = -0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2662);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-61);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2662);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2642);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-61);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2642);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2641);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-61);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2641);
    tri3_xyze(0, 1) = 1.01416e-15;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-61);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2679);
    tri3_xyze(0, 1) = 1.01416e-15;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-62);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01416e-15;
    tri3_xyze(1, 0) = -0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2641);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-62);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2641);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2659);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-62);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2662);
    tri3_xyze(0, 1) = 1.0615e-15;
    tri3_xyze(1, 1) = -0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2665);
    tri3_xyze(0, 2) = -0.00932385;
    tri3_xyze(1, 2) = -0.0708216;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-63);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0615e-15;
    tri3_xyze(1, 0) = -0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2665);
    tri3_xyze(0, 1) = -0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2645);
    tri3_xyze(0, 2) = -0.00932385;
    tri3_xyze(1, 2) = -0.0708216;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-63);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2645);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2642);
    tri3_xyze(0, 2) = -0.00932385;
    tri3_xyze(1, 2) = -0.0708216;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-63);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2642);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2662);
    tri3_xyze(0, 2) = -0.00932385;
    tri3_xyze(1, 2) = -0.0708216;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-63);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0615e-15;
    tri3_xyze(1, 0) = -0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2665);
    tri3_xyze(0, 1) = 1.05834e-15;
    tri3_xyze(1, 1) = -0.115451;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2667);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-64);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05834e-15;
    tri3_xyze(1, 0) = -0.115451;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2667);
    tri3_xyze(0, 1) = -0.0298809;
    tri3_xyze(1, 1) = -0.111517;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2647);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-64);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0298809;
    tri3_xyze(1, 0) = -0.111517;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2647);
    tri3_xyze(0, 1) = -0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2645);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-64);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2645);
    tri3_xyze(0, 1) = 1.0615e-15;
    tri3_xyze(1, 1) = -0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2665);
    tri3_xyze(0, 2) = -0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-64);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2682);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-71);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2682);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2662);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-71);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2662);
    tri3_xyze(0, 1) = 1.01416e-15;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-71);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01416e-15;
    tri3_xyze(1, 0) = -0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-71);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2699);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-72);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 1) = 1.01416e-15;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-72);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.01416e-15;
    tri3_xyze(1, 0) = -0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2679);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-72);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0218829;
    tri3_xyze(1, 0) = -0.0816682;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2685);
    tri3_xyze(0, 1) = 1.0615e-15;
    tri3_xyze(1, 1) = -0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2665);
    tri3_xyze(0, 2) = 0.00932385;
    tri3_xyze(1, 2) = -0.0708216;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-73);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0615e-15;
    tri3_xyze(1, 0) = -0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2665);
    tri3_xyze(0, 1) = 1.03009e-15;
    tri3_xyze(1, 1) = -0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2662);
    tri3_xyze(0, 2) = 0.00932385;
    tri3_xyze(1, 2) = -0.0708216;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-73);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.03009e-15;
    tri3_xyze(1, 0) = -0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2662);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2682);
    tri3_xyze(0, 2) = 0.00932385;
    tri3_xyze(1, 2) = -0.0708216;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-73);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05834e-15;
    tri3_xyze(1, 0) = -0.115451;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2667);
    tri3_xyze(0, 1) = 1.0615e-15;
    tri3_xyze(1, 1) = -0.0845492;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2665);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-74);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.0615e-15;
    tri3_xyze(1, 0) = -0.0845492;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2665);
    tri3_xyze(0, 1) = 0.0218829;
    tri3_xyze(1, 1) = -0.0816682;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2685);
    tri3_xyze(0, 2) = 0.012941;
    tri3_xyze(1, 2) = -0.0982963;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-74);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2702);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-81);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2682);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-81);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-81);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2719);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-82);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-82);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2699);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-82);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-91);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2702);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-91);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2702);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-91);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-91);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-92);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-92);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2719);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-92);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2725);
    tri3_xyze(0, 2) = 0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-93);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2702);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 2) = 0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-93);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2725);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-111);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-111);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-111);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-112);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-112);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-112);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-121);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-121);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-121);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-122);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-122);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-122);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-131);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3001);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-131);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3001);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-132);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-132);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2801);
    tri3_xyze(0, 1) = -0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2802);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-141);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2822);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2821);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-141);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2821);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2801);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-141);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2819);
    tri3_xyze(0, 1) = -0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2801);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-142);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2801);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2821);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-142);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2821);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2839);
    tri3_xyze(0, 2) = -0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-142);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2821);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2822);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2822);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2842);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2842);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = 0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2841);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = 0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2841);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2821);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-151);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2839);
    tri3_xyze(0, 1) = -0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2821);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2821);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = 0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2841);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = 0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2841);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2859);
    tri3_xyze(0, 2) = -0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-152);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0597853;
    tri3_xyze(1, 0) = 0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2845);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2842);
    tri3_xyze(0, 2) = -0.0566714;
    tri3_xyze(1, 2) = 0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2842);
    tri3_xyze(0, 1) = -0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2822);
    tri3_xyze(0, 2) = -0.0566714;
    tri3_xyze(1, 2) = 0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-153);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = 0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2841);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2842);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = 0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-161);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2842);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = 0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2862);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = 0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-161);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = 0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2862);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = 0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2861);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = 0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-161);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = 0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2861);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = 0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2841);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = 0.0430838;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-161);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2859);
    tri3_xyze(0, 1) = -0.0353553;
    tri3_xyze(1, 1) = 0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2841);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = 0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-162);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0353553;
    tri3_xyze(1, 0) = 0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2841);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = 0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2861);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = 0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-162);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = 0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2861);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = 0.0515711;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2879);
    tri3_xyze(0, 2) = -0.0330594;
    tri3_xyze(1, 2) = 0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-162);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2842);
    tri3_xyze(0, 1) = -0.0597853;
    tri3_xyze(1, 1) = 0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2845);
    tri3_xyze(0, 2) = -0.0434855;
    tri3_xyze(1, 2) = 0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-163);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = 0.0515711;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2862);
    tri3_xyze(0, 1) = -0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2842);
    tri3_xyze(0, 2) = -0.0434855;
    tri3_xyze(1, 2) = 0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-163);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = 0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2861);
    tri3_xyze(0, 1) = -0.0297746;
    tri3_xyze(1, 1) = 0.0515711;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2862);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = 0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-171);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = 0.0575201;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2882);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2881);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = 0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-171);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2881);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = 0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2861);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = 0.0501722;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-171);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0297746;
    tri3_xyze(1, 0) = 0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2879);
    tri3_xyze(0, 1) = -0.025;
    tri3_xyze(1, 1) = 0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2861);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = 0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-172);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.025;
    tri3_xyze(1, 0) = 0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2861);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2881);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = 0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-172);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2881);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = 0.0575201;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2899);
    tri3_xyze(0, 2) = -0.020782;
    tri3_xyze(1, 2) = 0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-172);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2881);
    tri3_xyze(0, 1) = -0.0154125;
    tri3_xyze(1, 1) = 0.0575201;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2882);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-181);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.05654e-15;
    tri3_xyze(1, 0) = 0.0595492;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2902);
    tri3_xyze(0, 1) = 9.90815e-16;
    tri3_xyze(1, 1) = 0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2901);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-181);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.90815e-16;
    tri3_xyze(1, 0) = 0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2901);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2881);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-181);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.0154125;
    tri3_xyze(1, 0) = 0.0575201;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2899);
    tri3_xyze(0, 1) = -0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2881);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-182);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = -0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2881);
    tri3_xyze(0, 1) = 9.90815e-16;
    tri3_xyze(1, 1) = 0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2901);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-182);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.90815e-16;
    tri3_xyze(1, 0) = 0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2901);
    tri3_xyze(0, 1) = 1.10943e-15;
    tri3_xyze(1, 1) = 0.0595492;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2919);
    tri3_xyze(0, 2) = -0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-182);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.90815e-16;
    tri3_xyze(1, 0) = 0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2901);
    tri3_xyze(0, 1) = 1.05654e-15;
    tri3_xyze(1, 1) = 0.0595492;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2902);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-191);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = 0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2921);
    tri3_xyze(0, 1) = 9.90815e-16;
    tri3_xyze(1, 1) = 0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2901);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-191);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.10943e-15;
    tri3_xyze(1, 0) = 0.0595492;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2919);
    tri3_xyze(0, 1) = 9.90815e-16;
    tri3_xyze(1, 1) = 0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2901);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-192);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 9.90815e-16;
    tri3_xyze(1, 0) = 0.05;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2901);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = 0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2921);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = 0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-192);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.05;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1863);
    hex8_xyze(0, 1) = 0.05;
    hex8_xyze(1, 1) = -1.73472e-18;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1864);
    hex8_xyze(0, 2) = 0;
    hex8_xyze(1, 2) = 0;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1875);
    hex8_xyze(0, 3) = 0;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1874);
    hex8_xyze(0, 4) = 0.05;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1984);
    hex8_xyze(0, 5) = 0.05;
    hex8_xyze(1, 5) = -2.08167e-18;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1985);
    hex8_xyze(0, 6) = 0;
    hex8_xyze(1, 6) = 0;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(1996);
    hex8_xyze(0, 7) = 0;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(1995);

    intersection.add_element(1880, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0;
    hex8_xyze(1, 0) = -0.1;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1873);
    hex8_xyze(0, 1) = 0;
    hex8_xyze(1, 1) = -0.05;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1874);
    hex8_xyze(0, 2) = -0.05;
    hex8_xyze(1, 2) = -0.05;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1885);
    hex8_xyze(0, 3) = -0.05;
    hex8_xyze(1, 3) = -0.1;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1884);
    hex8_xyze(0, 4) = 0;
    hex8_xyze(1, 4) = -0.1;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1994);
    hex8_xyze(0, 5) = 0;
    hex8_xyze(1, 5) = -0.05;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1995);
    hex8_xyze(0, 6) = -0.05;
    hex8_xyze(1, 6) = -0.05;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(2006);
    hex8_xyze(0, 7) = -0.05;
    hex8_xyze(1, 7) = -0.1;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(2005);

    intersection.add_element(1889, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1874);
    hex8_xyze(0, 1) = 0;
    hex8_xyze(1, 1) = 0;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1875);
    hex8_xyze(0, 2) = -0.05;
    hex8_xyze(1, 2) = 0;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1886);
    hex8_xyze(0, 3) = -0.05;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1885);
    hex8_xyze(0, 4) = 0;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1995);
    hex8_xyze(0, 5) = 0;
    hex8_xyze(1, 5) = 0;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1996);
    hex8_xyze(0, 6) = -0.05;
    hex8_xyze(1, 6) = 0;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(2007);
    hex8_xyze(0, 7) = -0.05;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(2006);

    intersection.add_element(1890, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0;
    hex8_xyze(1, 0) = 0;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1875);
    hex8_xyze(0, 1) = 0;
    hex8_xyze(1, 1) = 0.05;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1876);
    hex8_xyze(0, 2) = -0.05;
    hex8_xyze(1, 2) = 0.05;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1887);
    hex8_xyze(0, 3) = -0.05;
    hex8_xyze(1, 3) = 0;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1886);
    hex8_xyze(0, 4) = 0;
    hex8_xyze(1, 4) = 0;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(1996);
    hex8_xyze(0, 5) = 0;
    hex8_xyze(1, 5) = 0.05;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(1997);
    hex8_xyze(0, 6) = -0.05;
    hex8_xyze(1, 6) = 0.05;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(2008);
    hex8_xyze(0, 7) = -0.05;
    hex8_xyze(1, 7) = 0;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(2007);

    intersection.add_element(1891, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = -0.05;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.75;
    nids.push_back(1885);
    hex8_xyze(0, 1) = -0.05;
    hex8_xyze(1, 1) = 0;
    hex8_xyze(2, 1) = 0.75;
    nids.push_back(1886);
    hex8_xyze(0, 2) = -0.1;
    hex8_xyze(1, 2) = 0;
    hex8_xyze(2, 2) = 0.75;
    nids.push_back(1897);
    hex8_xyze(0, 3) = -0.1;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.75;
    nids.push_back(1896);
    hex8_xyze(0, 4) = -0.05;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.8;
    nids.push_back(2006);
    hex8_xyze(0, 5) = -0.05;
    hex8_xyze(1, 5) = 0;
    hex8_xyze(2, 5) = 0.8;
    nids.push_back(2007);
    hex8_xyze(0, 6) = -0.1;
    hex8_xyze(1, 6) = 0;
    hex8_xyze(2, 6) = 0.8;
    nids.push_back(2018);
    hex8_xyze(0, 7) = -0.1;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.8;
    nids.push_back(2017);

    intersection.add_element(1900, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.8;
    nids.push_back(1995);
    hex8_xyze(0, 1) = 0;
    hex8_xyze(1, 1) = 0;
    hex8_xyze(2, 1) = 0.8;
    nids.push_back(1996);
    hex8_xyze(0, 2) = -0.05;
    hex8_xyze(1, 2) = 0;
    hex8_xyze(2, 2) = 0.8;
    nids.push_back(2007);
    hex8_xyze(0, 3) = -0.05;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.8;
    nids.push_back(2006);
    hex8_xyze(0, 4) = 0;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.85;
    nids.push_back(2116);
    hex8_xyze(0, 5) = 0;
    hex8_xyze(1, 5) = 0;
    hex8_xyze(2, 5) = 0.85;
    nids.push_back(2117);
    hex8_xyze(0, 6) = -0.05;
    hex8_xyze(1, 6) = 0;
    hex8_xyze(2, 6) = 0.85;
    nids.push_back(2128);
    hex8_xyze(0, 7) = -0.05;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.85;
    nids.push_back(2127);

    intersection.add_element(1990, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_mesh().create_side_ids_cut_test();
  intersection.normal_mesh().create_side_ids_all_cut_test();

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
