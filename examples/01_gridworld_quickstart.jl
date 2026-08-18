# Quickstart: solve a POMDPs.jl gridworld with DORASolvers.
#
# Run from the repository root with
#     julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#     julia --project=examples examples/01_gridworld_quickstart.jl
#
# DORA is an online solver in the style of MCTS.jl: `solve` converts the MDP
# into a stochastic shortest path problem and returns a planner; the Dijkstra
# oracle calls run lazily at decision time inside `action`.

using DORASolvers
using POMDPs
using POMDPModels
using POMDPTools
using Random

# A 10x10 gridworld with one goal cell and two hazard cells. The commanded
# move succeeds with probability 0.7 and slips laterally otherwise.
rewards = Dict(GWPos(4, 3) => -10.0, GWPos(4, 6) => -10.0,
               GWPos(9, 3) => 10.0)
mdp = SimpleGridWorld(size=(10, 10), rewards=rewards, tprob=0.7)

# With the defaults, DORASolver derives the SSP structure from the reward
# function: positive-reward states become the goal, negative-reward states
# become crashes, and each step costs one unit. `known_costs=true` (default)
# seeds the cost estimates from the model, so no training is needed.
planner = solve(DORASolver(start=GWPos(1, 1)), mdp)

s = GWPos(1, 1)
println("action at $s:  ", action(planner, s))
println("value  at $s:  ", value(planner, s))

# The planner works with the standard POMDPTools simulators.
r = simulate(RolloutSimulator(rng=MersenneTwister(1), max_steps=100), mdp, planner)
println("rollout reward: ", r)

# The planner's policy is near optimal on the tabularized model. The exact
# evaluation helpers are exported, and accept the TabularSSP behind the
# planner directly.
V, _ = optimal_value(planner.tab)
Jstar = V[planner.tab.start]
J = eval_policy(planner.tab, planner.pi)[planner.tab.start]
println("optimal cost $Jstar, planner cost $J  (gap ",
        round(100 * (J - Jstar) / Jstar, digits=3), "%)")
