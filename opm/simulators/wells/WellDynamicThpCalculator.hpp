#include <config.h>


#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <opm/input/eclipse/Schedule/Group/Group.hpp>
#include <opm/input/eclipse/Schedule/Network/ExtNetwork.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>

#include <opm/simulators/utils/DeferredLogger.hpp>
#include <opm/simulators/utils/DeferredLoggingErrorHelpers.hpp>
#include <opm/simulators/utils/ParallelCommunication.hpp>

#include <opm/simulators/wells/VFPProdProperties.hpp>
#include <opm/simulators/wells/WellState.hpp>
#include <opm/simulators/wells/GroupState.hpp>
#include <opm/common/TimingMacros.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <set>
#include <stack>
#include <stdexcept>

namespace Opm {

template<typename Scalar>
struct ChainNode {
    ChainNode(const Network::Node& node) :
        name(node.name()),
        efficiency(node.efficiency()),
        terminal_pressure(node.terminal_pressure())
    {}

    void update_rates(const std::vector<Scalar>& delta_rates, const Scalar child_efficiency) {
        rates = orig_rates;
        for (auto i=0*delta_rates.size(); i<delta_rates.size(); ++i) {
            rates[i] -= child_efficiency*delta_rates[i]; // -= due to sign difference between well rates and node rates
        }
    }

    void set_rates(const std::vector<Scalar>& new_rates) {
        this->orig_rates = new_rates;
    }

    std::string name;
    Scalar efficiency{1.0};

    std::optional<Scalar> terminal_pressure{std::nullopt};
    std::optional<int> vfp_table_upbranch{std::nullopt};
    Scalar alq_value_upbranch{0.0};

    std::vector<Scalar> orig_rates{3, 0.0};
    std::vector<Scalar> rates{3, 0.0};
};

template<class Scalar>
class WellDynamicThpCalculator {
    public:
    WellDynamicThpCalculator() {}

    WellDynamicThpCalculator(const std::string& wname, const Schedule& schedule, const int report_step, const VFPProdProperties<Scalar>* vfpprops)
        : vfp_prod_props(vfpprops)
    {
        const auto& gname = schedule.getWell(wname, report_step).groupName();
        const auto& network = schedule[report_step].network();
        auto node_name = gname;
        assert(network.has_node(node_name));

        chain_nodes.emplace_back(network.node(node_name));
        auto* chain_node_p = &(chain_nodes.back());
        while (const auto branch_opt = network.uptree_branch(node_name)) {
            const auto& branch = *branch_opt;
            chain_node_p->vfp_table_upbranch = branch.vfp_table();
            chain_node_p->alq_value_upbranch = branch.alq_value().value_or(0.0);
            node_name = branch.uptree_node();
            const auto& uptree_node = network.node(node_name);
            chain_nodes.emplace_back(uptree_node);
            chain_node_p = &(chain_nodes.back());
            if (uptree_node.terminal_pressure()) {
                chain_node_p->terminal_pressure = uptree_node.terminal_pressure();
                break;
            }
        }
        assert(chain_node_p->terminal_pressure);
    }

    void set_rates(const std::map<std::string, const std::vector<Scalar>>& node_inflows, const std::vector<Scalar>& well_rates) {
        this->initialized_ = true;
        this->orig_well_rates = well_rates;
        for (auto& node : chain_nodes) {
            assert(node_inflows.count(node.name) > 0);
            node.set_rates(node_inflows.find(node.name)->second);
        }
    }

    Scalar operator()(const std::vector<Scalar>& rates) const {
        // Compute delta rates for current well (rates are well phase rates, i.e. should exclude any added gas lift)
        std::vector<Scalar> delta_rates(rates);
        for (auto i=0*rates.size(); i<rates.size(); ++i) {
            delta_rates[i] -= this->orig_well_rates[i];
        }

        // Update rates up the chain
        for (auto i=0*chain_nodes.size(); i<chain_nodes.size(); ++i) {
            chain_nodes[i].update_rates(delta_rates, i==0 ? 1.0 : chain_nodes[i-1].efficiency);
        }

        // Return dummy value for uninitialized (for now, avoid using optionals maybe?)
        if (chain_nodes.empty()) return -999999.0; 

        // Calculate pressure down the chain
        const auto lastidx = chain_nodes.size()-1;

        assert(chain_nodes[lastidx].terminal_pressure);
        auto node_pressure = chain_nodes[lastidx].terminal_pressure.value();
        for (auto i=lastidx; i>0; --i) {
            const auto vfp_table = chain_nodes[i-1].vfp_table_upbranch;
            if (vfp_table) {
                auto vfp_rates = chain_nodes[i-1].rates;
                std::transform(vfp_rates.begin(), vfp_rates.end(), vfp_rates.begin(), [](const auto r) { return -r; });
                node_pressure = vfp_prod_props->bhp(*vfp_table,
                                                    vfp_rates[BlackoilPhases::Aqua],
                                                    vfp_rates[BlackoilPhases::Liquid],
                                                    vfp_rates[BlackoilPhases::Vapour],
                                                    node_pressure,
                                                    chain_nodes[i-1].alq_value_upbranch,
                                                    0.0, //explicit_wfr
                                                    0.0, //explicit_gfr
                                                    false); //use_expvfp we dont support explicit lookup
            }
        }
        return node_pressure;
    }
    bool initialized() const { return initialized_; }

    private:
        mutable std::vector<ChainNode<Scalar>> chain_nodes;  // Chain of nodes from leaf group to root
        mutable std::vector<Scalar> orig_well_rates;
        const VFPProdProperties<Scalar>* vfp_prod_props {nullptr};
        bool initialized_ {false};
};

} // Namespace Opm