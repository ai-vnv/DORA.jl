# Online learning: DORA without known step costs.
#
# Run from the repository root with
#     julia --project=examples examples/02_online_learning.jl
#
# DORA's premise is that the transition kernel structure is known but the
# traversal costs are not: they are learned from observation, one confidence
# bound per state-action pair, with a fixed number of Dijkstra oracle calls
# per episode and no estimated transition kernel.

using DORA
using POMDPs
using POMDPModels

rewards = Dict(GWPos(4, 3) => -10.0, GWPos(4, 6) => -10.0,
               GWPos(9, 3) => 10.0)
mdp = SimpleGridWorld(size=(10, 10), rewards=rewards, tprob=0.7)

# Mode 1: self-simulated training inside solve() -------------------------
# The model itself is the generative simulator; observed step costs are
# sampled with noise (the protocol of experiment 4 in the paper).
planner = solve(DORASolver(start=GWPos(1, 1), known_costs=false,
                           train_episodes=300, cost_noise=0.2, seed=1), mdp)

V, _ = DORA.TabularSSPs.optimal_value(planner.tab)
Jstar = V[planner.tab.start]
J = DORA.TabularSSPs.eval_policy(planner.tab, replan!(planner))[planner.tab.start]
println("after 300 episodes: gap ", round(100 * (J - Jstar) / Jstar, digits=2), "%")

# training can continue at any time
train!(planner, 100)
J = DORA.TabularSSPs.eval_policy(planner.tab, replan!(planner))[planner.tab.start]
println("after 400 episodes: gap ", round(100 * (J - Jstar) / Jstar, digits=2), "%")

# Mode 2: deployment-time cost learning ----------------------------------
# Feed real observed costs back with observe!; the planner lazily replans
# on the next action query.
planner2 = solve(DORASolver(start=GWPos(1, 1), known_costs=false), mdp)
s = GWPos(1, 1)
a = action(planner2, s)             # plans on first query
observe!(planner2, s, a, 1.4)       # report the cost actually incurred
a2 = action(planner2, s)            # replans with the new information
println("oracle calls so far: ", oracle_calls(planner2.learner))
