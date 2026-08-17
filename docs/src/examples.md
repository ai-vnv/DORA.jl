# Examples

Four worked examples live in the repository's
[`examples/`](https://github.com/ai-vnv/DORA.jl/tree/main/examples)
directory. Run them from the repository root with

```bash
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/01_gridworld_quickstart.jl
```

## 1. Gridworld quickstart

[`01_gridworld_quickstart.jl`](https://github.com/ai-vnv/DORA.jl/blob/main/examples/01_gridworld_quickstart.jl)
solves a `SimpleGridWorld` from POMDPModels.jl with the solver defaults,
queries `action` and `value`, runs the planner under
`POMDPTools.RolloutSimulator`, and compares the planner's policy cost to the
optimal cost of the tabularized model (the gap is below 0.01%).

## 2. Online cost learning

[`02_online_learning.jl`](https://github.com/ai-vnv/DORA.jl/blob/main/examples/02_online_learning.jl)
turns the known-costs seeding off and learns traversal costs from noisy
observations, in both modes: self-simulated training inside `solve` (plus
`train!` to continue), and deployment-time learning through `observe!` with
lazy replanning.

## 3. Warehouse navigation benchmark

[`03_warehouse_navigation.jl`](https://github.com/ai-vnv/DORA.jl/blob/main/examples/03_warehouse_navigation.jl)
builds the paper's warehouse domain — a shelf grid with unknown terrain
costs, lateral actuation slip, and a hazard band of dynamic obstacles — and
drives the learner with the episode-level API (`plan!`, `run_episode!`),
tracking regret and planner work. It also demonstrates the
chance-constrained variant `RiskDORA`, whose risk multiplier is updated by
projected dual ascent on the realized contact indicator.

## 4. Custom MDP with explicit SSP structure

[`04_custom_mdp.jl`](https://github.com/ai-vnv/DORA.jl/blob/main/examples/04_custom_mdp.jl)
defines a small river-crossing MDP and shows how to pass `classify`, `cost`,
`start`, and the penalty scales explicitly when the reward-based defaults do
not fit — for example, action-dependent costs where a careful move is slow
but safe and a long jump is fast but risky.
