"""
DORA.jl. Dijkstra Oracle Reduced-cost Algorithm for online stochastic shortest
path navigation, packaged as a solver for models formulated with POMDPs.jl.

Quick start:

    using DORA, POMDPs, POMDPModels
    planner = solve(DORASolver(), SimpleGridWorld())
    a = action(planner, GWPos(1, 1))

See the `examples/` directory for worked examples, and the research
repository (mansurarief/dora-dijkstra-mdp) for the paper and the scripts
that reproduce its experiments.
"""
module DORA

include("RNG.jl")
include("Dijkstra.jl")
include("NavSSP.jl")
include("TabularSSP.jl")
include("Learners.jl")

using .RNGs
using .Dijkstra
using .NavSSPs
using .TabularSSPs
using .Learners

include("solver.jl")

export SplitMix64, rand01, randint, uniform, categorical
export ReverseAdj, dijkstra_policy, dijkstra_to_goal, reset_ops!, edge_scans
export NavSSP, build, eval_policy, optimal_value, outcome_rates,
       causality_margin, reduced_costs, NACT, MAXOUT
export TabularSSP, tabularize, TabularSSPs
export DORALearner, DORA0, CED, EGD, OptimisticVI, RiskDORA, Sarsa, MCTSPlan,
       plan!, observe!, run_episode!, update_multiplier!, work,
       oracle_calls, vi_sweeps, rhat
export DORASolver, DORAPlanner, replan!, train!

end # module
