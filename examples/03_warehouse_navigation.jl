# The warehouse navigation benchmark and the low-level learner API.
#
# Run from the repository root with
#     julia --project=examples examples/03_warehouse_navigation.jl
#
# DORASolvers.jl ships the warehouse SSP used as the paper's primary benchmark: a
# shelf grid with unknown terrain costs, lateral actuation slip, and a hazard
# band of dynamic obstacles. It exposes the POMDPs.jl MDP interface, and the
# low-level learner API gives episode-by-episode control.

using DORASolvers
using POMDPs

# 20x20 warehouse; the SplitMix64 terrain seed reproduces the paper maps.
m = build(; n=20, eps=0.10, p_haz=0.14, rough=0.5, c_crash=80.0, c_to=60.0,
          horizon=150, cost_noise=0.12, terrain_rng=SplitMix64(7000))

V, pistar = optimal_value(m)
sr, cr, tr = outcome_rates(m, pistar)
println("states: ", m.NS, "  optimal cost: ", round(V[m.start], digits=2),
        "  success rate: ", round(sr, digits=3))

# Online learning with the episode-level API: plan, act, observe, repeat.
K = 200
L = DORALearner(m, K)
rng = SplitMix64(31000)
total_regret = 0.0
for k in 1:K
    pi = plan!(L)                       # a fixed number of Dijkstra calls
    J = eval_policy(m, pi)[m.start]     # exact cost of the current policy
    global total_regret += max(0.0, J - V[m.start])
    run_episode!(L, pi, rng)            # simulate and feed observations back
end
println("cumulative regret after $K episodes: ", round(total_regret, digits=1))
println("planner work per episode: ", round(work(L) / K, digits=0), " edge scans")

# Chance-constrained variant: bound the contact probability by a budget.
R = RiskDORA(m, K; budget=0.05)
for k in 1:50
    pi = plan!(R)
    _, _, crashed, _ = run_episode!(R, pi, rng)
    update_multiplier!(R, crashed)      # projected dual ascent
end
println("risk multiplier after 50 episodes: ", round(R.lam, digits=2))
