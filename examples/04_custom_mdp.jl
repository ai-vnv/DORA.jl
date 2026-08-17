# Using DORA on your own POMDPs.jl model, with explicit SSP structure.
#
# Run from the repository root with
#     julia --project=examples examples/04_custom_mdp.jl
#
# DORASolver needs to know which states are goals, which are failures, and
# what a step costs. For models where the reward-based defaults do not apply,
# pass `classify`, `cost`, `start`, and (for state types that do not hash by
# value) `key` explicitly.

using DORASolvers
using POMDPs
using POMDPTools

# A small river-crossing MDP: move right along a 1D chain of stones, but each
# hop can slip back. Stone 6 is the far bank (goal); stone 0 is the water
# (failure). Actions: 1 = careful hop (slow, safe), 2 = long jump (fast,
# risky).
struct RiverMDP <: MDP{Int,Int} end
POMDPs.actions(::RiverMDP) = [1, 2]
POMDPs.discount(::RiverMDP) = 1.0
POMDPs.isterminal(::RiverMDP, s::Int) = s == 0 || s >= 6
POMDPs.initialstate(::RiverMDP) = Deterministic(1)
function POMDPs.transition(::RiverMDP, s::Int, a::Int)
    (s == 0 || s >= 6) && return Deterministic(s)
    if a == 1   # careful hop: +1 w.p. 0.9, stay w.p. 0.1
        return SparseCat([min(s + 1, 6), s], [0.9, 0.1])
    else        # long jump: +2 w.p. 0.6, fall in w.p. 0.4
        return SparseCat([min(s + 2, 6), 0], [0.6, 0.4])
    end
end
# reward is not needed: we specify costs directly below

planner = solve(DORASolver(
        start = 1,
        classify = sp -> sp >= 6 ? :goal : sp == 0 ? :crash : :normal,
        cost = (s, a, sp) -> a == 1 ? 2.0 : 1.0,   # careful hops take longer
        c_to = 60.0,        # penalty when the horizon runs out
        c_crash = 30.0,     # penalty for falling in
        horizon = 40,
    ), RiverMDP())

for s in 1:5
    println("stone $s: ", action(planner, s) == 1 ? "careful hop" : "long jump",
            "  (cost-to-goal ", round(-value(planner, s), digits=2), ")")
end

# The tabularized SSP is available for exact analysis through the same
# exported helpers that the warehouse domain uses.
V, pistar = optimal_value(planner.tab)
sr, cr, _ = outcome_rates(planner.tab, pistar)
println("optimal expected cost: ", round(V[planner.tab.start], digits=2),
        "  success rate: ", round(sr, digits=3))
