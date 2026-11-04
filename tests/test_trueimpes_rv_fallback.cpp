/*
  Copyright 2026 Equinor ASA

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"

#define BOOST_TEST_MODULE TrueImpesRvFallback

#include "SimulatorFixture.hpp"

#include <opm/grid/utility/ElementChunks.hpp>
#include <opm/simulators/flow/FlowProblemBlackoil.hpp>
#include <opm/simulators/linalg/getQuasiImpesWeights.hpp>
#include <opm/simulators/wells/BlackoilWellModel.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace
{

using TypeTag = Opm::Properties::TTag::TestTypeTag;
using Simulator = Opm::GetPropType<TypeTag, Opm::Properties::Simulator>;
using ElementContext = Opm::GetPropType<TypeTag, Opm::Properties::ElementContext>;
using GridView = Opm::GetPropType<TypeTag, Opm::Properties::GridView>;
using Indices = Opm::GetPropType<TypeTag, Opm::Properties::Indices>;
using FluidSystem = Opm::GetPropType<TypeTag, Opm::Properties::FluidSystem>;
using PrimaryVariables = Opm::GetPropType<TypeTag, Opm::Properties::PrimaryVariables>;
using Vector = Opm::GetPropType<TypeTag, Opm::Properties::GlobalEqVector>;

constexpr int pressureIndex = Indices::pressureSwitchIdx;

Vector trueImpesWeights(const Simulator& simulator, const double rvFallbackThreshold)
{
    Vector weights(simulator.model().numGridDof());
    const ElementContext elemCtx(simulator);
    const Opm::ElementChunks<GridView, Dune::Partitions::All>
        chunks(simulator.gridView(), Dune::Partitions::all, /*num_chunks=*/1);
    Opm::Amg::getTrueImpesWeights(pressureIndex, weights, elemCtx, simulator.model(),
                                  chunks, /*enable_thread_parallel=*/false, rvFallbackThreshold);
    return weights;
}

Vector normalisedAnalyticWeights(const Simulator& simulator)
{
    Vector weights(simulator.model().numGridDof());
    const ElementContext elemCtx(simulator);
    const Opm::ElementChunks<GridView, Dune::Partitions::All>
        chunks(simulator.gridView(), Dune::Partitions::all, /*num_chunks=*/1);
    Opm::Amg::getTrueImpesWeightsAnalytic(pressureIndex, weights, elemCtx, simulator.model(),
                                          chunks, /*enable_thread_parallel=*/false);
    for (auto& block : weights) {
        double maxAbs = 0.0;
        for (const double w : block) {
            maxAbs = std::max(maxAbs, std::abs(w));
        }
        block /= maxAbs;
    }
    return weights;
}

unsigned gasEqIdx()
{
    return Indices::conti0EqIdx + FluidSystem::canonicalToActiveCompIdx(FluidSystem::gasCompIdx);
}

unsigned oilEqIdx()
{
    return Indices::conti0EqIdx + FluidSystem::canonicalToActiveCompIdx(FluidSystem::oilCompIdx);
}

} // namespace

using SimulatorFixture = Opm::SimulatorFixture;
BOOST_GLOBAL_FIXTURE(SimulatorFixture);

BOOST_AUTO_TEST_CASE(RvCellsWithBgVeryStrongDependenceOnRvUseAnalyticWeights)
{
    const auto simulator = Opm::initSimulator<TypeTag>("trueimpes_rv_fallback.DATA",
                                                       "test_trueimpes_rv_fallback");
    simulator->model().applyInitialSolution();
    simulator->setTimeStepSize(86400.0);

    const auto numDof = simulator->model().numGridDof();
    BOOST_REQUIRE_GT(numDof, 0u);

    const auto numerical = trueImpesWeights(*simulator, /*rvFallbackThreshold=*/0.0);
    const auto withFallback = trueImpesWeights(*simulator, /*rvFallbackThreshold=*/1.0e4);
    const auto analytic = normalisedAnalyticWeights(*simulator);

    int numSteepRvCells = 0;
    int numRegularRvCells = 0;
    for (std::size_t cell = 0; cell < numDof; ++cell) {
        const bool isRvCell = simulator->model().solution(/*timeIdx=*/0)[cell].primaryVarsMeaningGas()
            == PrimaryVariables::GasMeaning::Rv;
        const bool steepRegion = simulator->problem().pvtRegionIndex(cell) == 0;
        BOOST_TEST_CONTEXT("cell " << cell << ", Rv cell " << isRvCell
                           << ", steep PVTG region " << steepRegion)
        {
            for (const auto& block : {numerical[cell], withFallback[cell], analytic[cell]}) {
                for (const double w : block) {
                    BOOST_REQUIRE(std::isfinite(w));
                }
            }
            if (isRvCell && steepRegion) {
                ++numSteepRvCells;
                // The numerical weights have a negligible gas weight, so the
                // pressure equation is essentially the vaporised-oil equation.
                BOOST_CHECK_LT(std::abs(numerical[cell][gasEqIdx()]),
                               1.0e-3 * std::abs(numerical[cell][oilEqIdx()]));
                for (int eq = 0; eq < Indices::numEq; ++eq) {
                    BOOST_CHECK_CLOSE(withFallback[cell][eq], analytic[cell][eq], 1.0e-10);
                }
            }
            else {
                numRegularRvCells += isRvCell ? 1 : 0;
                for (int eq = 0; eq < Indices::numEq; ++eq) {
                    BOOST_CHECK_EQUAL(withFallback[cell][eq], numerical[cell][eq]);
                }
            }
        }
    }

    // Both gas caps must be represented with Rv as primary variable,
    // otherwise the test is vacuous.
    BOOST_CHECK_GT(numSteepRvCells, 0);
    BOOST_CHECK_GT(numRegularRvCells, 0);
}
