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
#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/simulators/linalg/TrueImpesWeightsOptions.hpp>
#include <opm/simulators/linalg/getQuasiImpesWeights.hpp>
#include <opm/simulators/linalg/setupPropertyTree.hpp>
#include <opm/simulators/wells/BlackoilWellModel.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

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
using Options = Opm::Amg::TrueImpesWeightsOptions;

constexpr int pressureIndex = Indices::pressureSwitchIdx;

std::unique_ptr<Simulator> initialisedSimulator()
{
    auto simulator = Opm::initSimulator<TypeTag>("trueimpes_rv_fallback.DATA",
                                                 "test_trueimpes_rv_fallback");
    simulator->model().applyInitialSolution();
    simulator->setTimeStepSize(86400.0);
    BOOST_REQUIRE_GT(simulator->model().numGridDof(), 0u);
    return simulator;
}

Vector trueImpesWeights(const Simulator& simulator, const Options& options)
{
    Vector weights(simulator.model().numGridDof());
    const ElementContext elemCtx(simulator);
    const Opm::ElementChunks<GridView, Dune::Partitions::All>
        chunks(simulator.gridView(), Dune::Partitions::all, /*num_chunks=*/1);
    Opm::Amg::getTrueImpesWeights(pressureIndex, weights, elemCtx, simulator.model(),
                                  chunks, /*enable_thread_parallel=*/false, options);
    return weights;
}

Vector analyticWeights(const Simulator& simulator)
{
    Vector weights(simulator.model().numGridDof());
    const ElementContext elemCtx(simulator);
    const Opm::ElementChunks<GridView, Dune::Partitions::All>
        chunks(simulator.gridView(), Dune::Partitions::all, /*num_chunks=*/1);
    Opm::Amg::getTrueImpesWeightsAnalytic(pressureIndex, weights, elemCtx, simulator.model(),
                                          chunks, /*enable_thread_parallel=*/false);
    return weights;
}

Vector::block_type maxAbsNormalised(Vector::block_type block)
{
    double maxAbs = 0.0;
    for (const double w : block) {
        maxAbs = std::max(maxAbs, std::abs(w));
    }
    block /= maxAbs;
    return block;
}

// Weighted storage of the mass conservation equations, and porosity, per cell.
std::vector<std::array<double, 2>> weightedStorageAndPorosity(const Simulator& simulator,
                                                              const Vector& weights)
{
    std::vector<std::array<double, 2>> result(simulator.model().numGridDof());
    ElementContext elemCtx(simulator);
    using Evaluation = Opm::GetPropType<TypeTag, Opm::Properties::Evaluation>;
    Dune::FieldVector<Evaluation, Indices::numEq> storage;
    for (const auto& elem : elements(simulator.gridView())) {
        elemCtx.updatePrimaryStencil(elem);
        elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);
        const auto cell = elemCtx.globalSpaceIndex(/*spaceIdx=*/0, /*timeIdx=*/0);
        simulator.model().localLinearizer(0).localResidual()
            .computeStorage(storage, elemCtx, /*spaceIdx=*/0, /*timeIdx=*/0);
        double weighted = 0.0;
        for (int eq = Indices::conti0EqIdx; eq < Indices::conti0EqIdx + Indices::numPhases; ++eq) {
            weighted += weights[cell][eq] * storage[eq].value();
        }
        result[cell] = {weighted, Opm::getValue(elemCtx.intensiveQuantities(0, 0).porosity())};
    }
    return result;
}

bool isRvCell(const Simulator& simulator, const std::size_t cell)
{
    return simulator.model().solution(/*timeIdx=*/0)[cell].primaryVarsMeaningGas()
        == PrimaryVariables::GasMeaning::Rv;
}

// Region 1 of the deck has the PVTG table with a very strong Bg(Rv) dependence.
bool inSteepPvtgRegion(const Simulator& simulator, const std::size_t cell)
{
    return simulator.problem().pvtRegionIndex(cell) == 0;
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
    const auto simulator = initialisedSimulator();
    const auto numDof = simulator->model().numGridDof();

    const Options noFallback{Options::Scaling::MaxAbs, /*rv_fallback_threshold=*/0.0};
    const Options withFallback{Options::Scaling::MaxAbs, /*rv_fallback_threshold=*/1.0e4};
    const auto numerical = trueImpesWeights(*simulator, noFallback);
    const auto fallback = trueImpesWeights(*simulator, withFallback);
    const auto analytic = analyticWeights(*simulator);

    int numSteepRvCells = 0;
    int numRegularRvCells = 0;
    for (std::size_t cell = 0; cell < numDof; ++cell) {
        const bool rvCell = isRvCell(*simulator, cell);
        const bool steepRegion = inSteepPvtgRegion(*simulator, cell);
        BOOST_TEST_CONTEXT("cell " << cell << ", Rv cell " << rvCell
                           << ", steep PVTG region " << steepRegion)
        {
            for (const auto& block : {numerical[cell], fallback[cell], analytic[cell]}) {
                for (const double w : block) {
                    BOOST_REQUIRE(std::isfinite(w));
                }
            }
            if (rvCell && steepRegion) {
                ++numSteepRvCells;
                // The numerical weights have a negligible gas weight, so the
                // pressure equation is essentially the vaporised-oil equation.
                BOOST_CHECK_LT(std::abs(numerical[cell][gasEqIdx()]),
                               1.0e-3 * std::abs(numerical[cell][oilEqIdx()]));
                const auto expected = maxAbsNormalised(analytic[cell]);
                for (int eq = 0; eq < Indices::numEq; ++eq) {
                    BOOST_CHECK_CLOSE(fallback[cell][eq], expected[eq], 1.0e-10);
                }
            }
            else {
                numRegularRvCells += rvCell ? 1 : 0;
                for (int eq = 0; eq < Indices::numEq; ++eq) {
                    BOOST_CHECK_EQUAL(fallback[cell][eq], numerical[cell][eq]);
                }
            }
        }
    }

    // Both gas caps must be represented with Rv as primary variable,
    // otherwise the test is vacuous.
    BOOST_CHECK_GT(numSteepRvCells, 0);
    BOOST_CHECK_GT(numRegularRvCells, 0);
}

BOOST_AUTO_TEST_CASE(PoreVolumeScalingGivesReservoirVolumeUnits)
{
    const auto simulator = initialisedSimulator();
    const auto numDof = simulator->model().numGridDof();

    // The default options: pore volume scaling with the Rv fallback.
    const Options defaults{};
    BOOST_REQUIRE(defaults.scaling == Options::Scaling::PoreVolume);
    BOOST_REQUIRE_GT(defaults.rv_fallback_threshold, 0.0);

    const auto poreVolume = trueImpesWeights(*simulator, defaults);
    const auto maxAbs = trueImpesWeights(*simulator, Options{Options::Scaling::MaxAbs,
                                                             defaults.rv_fallback_threshold});
    const auto analytic = analyticWeights(*simulator);
    const auto storage = weightedStorageAndPorosity(*simulator, poreVolume);
    const auto analyticStorage = weightedStorageAndPorosity(*simulator, analytic);

    for (std::size_t cell = 0; cell < numDof; ++cell) {
        BOOST_TEST_CONTEXT("cell " << cell << ", Rv cell " << isRvCell(*simulator, cell)
                           << ", steep PVTG region " << inSteepPvtgRegion(*simulator, cell))
        {
            // Reservoir volume units: the weighted storage is the porosity, as
            // it is for the analytic weights.
            BOOST_CHECK_CLOSE(storage[cell][0], storage[cell][1], 1.0e-10);
            BOOST_CHECK_CLOSE(analyticStorage[cell][0], analyticStorage[cell][1], 1.0e-8);

            // Only the scale differs from the max-abs scaled weights.
            const auto normalised = maxAbsNormalised(poreVolume[cell]);
            for (int eq = 0; eq < Indices::numEq; ++eq) {
                BOOST_CHECK_CLOSE(normalised[eq], maxAbs[cell][eq], 1.0e-10);
            }

            // Fallback cells therefore get the analytic weights themselves.
            if (isRvCell(*simulator, cell) && inSteepPvtgRegion(*simulator, cell)) {
                for (int eq = 0; eq < Indices::numEq; ++eq) {
                    BOOST_CHECK_CLOSE(poreVolume[cell][eq], analytic[cell][eq], 1.0e-8);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(OptionsFromPropertyTree)
{
    const auto defaults = Options::fromPropertyTree(Opm::PropertyTree{});
    BOOST_CHECK(defaults.scaling == Options::Scaling::PoreVolume);
    BOOST_CHECK_EQUAL(defaults.rv_fallback_threshold, 1.0e4);

    Opm::PropertyTree prm;
    prm.put("weight_scaling", std::string{"max_abs"});
    prm.put("rv_fallback_threshold", 0.0);
    const auto options = Options::fromPropertyTree(prm);
    BOOST_CHECK(options.scaling == Options::Scaling::MaxAbs);
    BOOST_CHECK_EQUAL(options.rv_fallback_threshold, 0.0);

    // Round trip through put().
    Opm::PropertyTree stored;
    options.put(stored, "preconditioner.");
    const auto restored = Options::fromPropertyTree(stored.get_child("preconditioner"));
    BOOST_CHECK(restored.scaling == options.scaling);
    BOOST_CHECK_EQUAL(restored.rv_fallback_threshold, options.rv_fallback_threshold);

    Opm::PropertyTree bad;
    bad.put("weight_scaling", std::string{"unknown"});
    BOOST_CHECK_THROW(Options::fromPropertyTree(bad), std::invalid_argument);

    // The generated true-IMPES configurations state the defaults explicitly.
    Opm::FlowLinearSolverParameters parameters;
    for (const auto& tree : {Opm::setupCPRW("cprw", parameters),
                             Opm::setupCPR("cpr_trueimpes", parameters)}) {
        BOOST_CHECK_EQUAL(tree.get<std::string>("preconditioner.weight_scaling"), "pore_volume");
        BOOST_CHECK_EQUAL(tree.get<double>("preconditioner.rv_fallback_threshold"), 1.0e4);
    }
    const auto system = Opm::setupSystemCPR("system_cprw", parameters);
    BOOST_CHECK_EQUAL(system.get<std::string>(
                          "preconditioner.reservoir_solver.preconditioner.weight_scaling"),
                      "pore_volume");
}
