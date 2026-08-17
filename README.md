# DORA.jl

[![CI](https://github.com/ai-vnv/DORA.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ai-vnv/DORA.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/ai-vnv/DORA.jl/graph/badge.svg)](https://codecov.io/gh/ai-vnv/DORA.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai-vnv.github.io/DORA.jl/dev/)

DORA (Dijkstra Oracle Reduced-cost Algorithm) is an online solver for
stochastic shortest path problems specified using the
[POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

A one-pass label-setting method such as Dijkstra is usually dismissed for
stochastic shortest path problems because it is only exact when the value
function decreases along every positive-probability transition. The condition
that actually matters is weaker: if the *reduced costs*
`w*(s,a) = Q*(s,a) - V*(sigma(s,a))` are nonnegative, where `sigma(s,a)` is
the intended successor, then Dijkstra run with them as edge weights returns
exactly the optimal value function and policy. DORA estimates these weights
online, using one confidence bound per state-action pair and a fixed number
of Dijkstra oracle calls per episode, and never estimates a transition
kernel.

DORA is an online solver: `solve` converts the MDP into a
tabular SSP and returns a planner; the Dijkstra oracle calls run lazily at
decision time inside `action`.

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/ai-vnv/DORA.jl")
```

## Usage

```julia
using POMDPs
using DORA
using POMDPModels
using POMDPTools

rewards = Dict(GWPos(4, 3) => -10.0, GWPos(4, 6) => -10.0,
               GWPos(9, 3) => 10.0)
mdp = SimpleGridWorld(size=(10, 10), rewards=rewards, tprob=0.7)

solver = DORASolver(start=GWPos(1, 1))
planner = solve(solver, mdp)

a = action(planner, GWPos(1, 1))    # plans on first query
v = value(planner, GWPos(1, 1))     # negated learned cost-to-goal

r = simulate(RolloutSimulator(max_steps=100), mdp, planner)
```

By default (`known_costs=true`) the cost statistics are seeded from the
model's expected step costs, so the planner is usable immediately. To use
DORA as an online *learner* instead:

```julia
# self-simulated training on the model
planner = solve(DORASolver(start=GWPos(1, 1), known_costs=false,
                           train_episodes=400), mdp)
train!(planner, 100)                # continue training any time

# or deployment-time cost learning
observe!(planner, s, a, observed_cost)   # report a real observed cost
a = action(planner, s)                   # lazily replans
```

Solver options are controlled with keyword arguments to `DORASolver`; use
`?DORASolver` for the full list. The main ones:

- `iters::Int = 3`: Dijkstra oracle calls per (re)planning round
- `alpha::Float64 = 0.4`: damping of the cost-to-goal update
- `beta::Float64 = 0.05`: confidence radius scale for optimistic estimates
- `known_costs::Bool = true`: seed cost statistics from the model
- `train_episodes::Int = 0`: self-simulated training episodes inside `solve`
- `start`, `classify`, `cost`, `key`: SSP structure; derived from
  `initialstate`, `isterminal`, and the reward function by default
- `horizon`, `c_to`, `c_crash`, `c_min`, `step_cost`, `cost_noise`: episode
  horizon, timeout/crash penalties, and cost scale

Requirements: the model must have discrete states and actions and an explicit
`transition` distribution. DORA treats the model as an undiscounted SSP; a
discount below one is ignored with a warning.

## Examples

Worked examples are in [`examples/`](examples/):

1. [`01_gridworld_quickstart.jl`](examples/01_gridworld_quickstart.jl) —
   solve a POMDPs.jl gridworld and simulate the planner
2. [`02_online_learning.jl`](examples/02_online_learning.jl) — learn step
   costs online, by self-simulation and by deployment-time observation
3. [`03_warehouse_navigation.jl`](examples/03_warehouse_navigation.jl) — the
   paper's warehouse benchmark and the low-level episode API, including the
   chance-constrained variant
4. [`04_custom_mdp.jl`](examples/04_custom_mdp.jl) — a custom MDP with
   explicit goal/failure classification and action-dependent costs

Run them from the repository root with

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/01_gridworld_quickstart.jl
```

## Package contents

Beyond the solver, the package exports the building blocks used by the paper:

- `tabularize`: convert any discrete POMDPs.jl MDP into a tabular SSP with
  collapsed goal and crash sinks
- `DORALearner`, `plan!`, `observe!`, `run_episode!`: the episode-level
  online learning API, plus the baseline learners (`DORA0`, `CED`, `EGD`,
  `OptimisticVI`, `Sarsa`, `MCTSPlan`) and the chance-constrained
  `RiskDORA` with `update_multiplier!`
- `build`: the warehouse navigation benchmark domain (a POMDPs.jl MDP)
- `dijkstra_policy`, `ReverseAdj`, `edge_scans`: the backward Dijkstra
  oracle with its implementation-independent work counter
- `SplitMix64`: a deterministic RNG mirrored bit-for-bit by the Python
  reference implementation in the research repository

## Citation

If you use this package, please cite *Dijkstra as an Oracle for Online
Stochastic Shortest Path Navigation with Provable Guarantees* paper (TBD).
