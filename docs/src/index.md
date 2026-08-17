# DORASolvers.jl

DORA (Dijkstra Oracle Reduced-cost Algorithm) is an online solver for
stochastic shortest path (SSP) problems specified using the
[POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

## The idea

A one-pass label-setting method such as Dijkstra is usually dismissed for
stochastic shortest path problems because it is only exact when the value
function decreases along every positive-probability transition. The condition
that actually matters is weaker: if the *reduced costs*

```math
w^*(s,a) = Q^*(s,a) - V^*(\sigma(s,a))
```

are nonnegative, where ``\sigma(s,a)`` is the intended successor of action
``a`` in state ``s``, then Dijkstra run with them as edge weights returns
exactly the optimal value function and exactly the optimal policy. DORA
estimates these weights online, using one confidence bound per state-action
pair and a fixed number of Dijkstra oracle calls per episode, and never
estimates a transition kernel.

Like MCTS.jl, DORA follows the online-solver pattern of POMDPs.jl:
`solve` converts the MDP into a tabular SSP and returns a planner; the
Dijkstra oracle calls run lazily at decision time inside `action`.

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/ai-vnv/DORASolvers.jl")
```

## Quick start

```julia
using POMDPs
using DORASolvers
using POMDPModels
using POMDPTools

rewards = Dict(GWPos(4, 3) => -10.0, GWPos(4, 6) => -10.0,
               GWPos(9, 3) => 10.0)
mdp = SimpleGridWorld(size=(10, 10), rewards=rewards, tprob=0.7)

planner = solve(DORASolver(start=GWPos(1, 1)), mdp)

a = action(planner, GWPos(1, 1))    # plans on first query
v = value(planner, GWPos(1, 1))     # negated learned cost-to-goal

r = simulate(RolloutSimulator(max_steps=100), mdp, planner)
```

## Manual outline

```@contents
Pages = ["guide.md", "examples.md", "api.md"]
Depth = 2
```

## Citation

If you use this package, please cite *Dijkstra as an Oracle for Online
Stochastic Shortest Path Navigation with Provable Guarantees* (paper
forthcoming).
