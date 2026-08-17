# Solver Guide

## How a model becomes an SSP

`solve(::DORASolver, ::MDP)` first converts the model into a tabular
stochastic shortest path problem with [`tabularize`](@ref):

1. Starting from a single start state, all reachable states are enumerated
   through the model's `transition` distributions.
2. Every successor is classified as `:goal`, `:crash`, or `:normal`. Goal
   and crash states are collapsed into two absorbing sink indices.
3. Rewards are converted into positive traversal costs.
4. Each state-action pair is assigned a *determinized successor* — its most
   likely non-self, non-crash outcome — which defines the graph the Dijkstra
   oracle searches.

By default the SSP structure is derived from the model:

- **start**: sampled from `initialstate(mdp)`
- **classify**: terminal or positive-reward states become `:goal`,
  negative-reward states become `:crash` (reward sign taken as
  `maximum(reward(mdp, sp, a))` over actions)
- **cost**: `max(c_min, step_cost - reward(mdp, s, a))`, so unit steps cost
  `step_cost` and penalized steps cost more

For models where these heuristics do not apply, pass `start`, `classify`,
`cost`, and (for state types that do not hash by value) `key` explicitly —
see the [custom MDP example](examples.md).

!!! note "Requirements"
    The model must have discrete states and actions and an explicit
    `transition` distribution (something `POMDPTools.weighted_iterator` can
    enumerate). DORA treats the model as an undiscounted SSP; a discount
    below one is ignored with a warning.

## Three ways to use the planner

**Decision-time planning (default).** With `known_costs=true`, the cost
statistics are seeded from the model's expected step costs, so the planner is
usable immediately with any standard simulator. Planning is lazy: the
Dijkstra oracle calls run on the first `action` query and are cached until
new information arrives.

```julia
planner = solve(DORASolver(start=s0), mdp)
a = action(planner, s)
```

**Self-simulated training.** With `known_costs=false`, the traversal costs
are treated as unknown and learned from noisy observations, using the model
itself as the generative simulator:

```julia
planner = solve(DORASolver(start=s0, known_costs=false,
                           train_episodes=400), mdp)
train!(planner, 100)          # continue training any time
```

**Deployment-time cost learning.** Report real observed costs with
`observe!`; the planner lazily replans on the next `action` query:

```julia
observe!(planner, s, a, observed_cost)
a = action(planner, s)        # replans first
```

## Solver options

| Option | Default | Meaning |
|---|---|---|
| `iters` | `3` | Dijkstra oracle calls per (re)planning round |
| `alpha` | `0.4` | damping of the cost-to-goal update |
| `beta` | `0.05` | confidence radius scale for optimistic cost estimates |
| `optimistic` | `true` | subtract the confidence radius from cost estimates |
| `correct` | `true` | apply the reduced-cost drift correction (disable to get plain determinize-and-replan) |
| `explore_eps` | `0.0` | epsilon-greedy exploration during training episodes |
| `known_costs` | `true` | seed cost statistics from the model |
| `train_episodes` | `0` | self-simulated training episodes inside `solve` |
| `episodes_budget` | `500` | horizon ``K`` in the confidence radius schedule |
| `seed` | `0` | seed of the internal `SplitMix64` stream |
| `start` | derived | initial state |
| `classify` | derived | `sp -> :goal / :crash / :normal` |
| `cost` | derived | `(s, a, sp) -> Float64` traversal cost |
| `key` | `identity` | hashable state identifier |
| `step_cost` | `1.0` | base cost of one step under the default `cost` |
| `c_min` | `0.25` | lower bound on any traversal cost |
| `horizon` | `200` | episode horizon |
| `c_to` | `2 * horizon * step_cost` | timeout penalty |
| `c_crash` | `horizon * step_cost` | crash penalty |
| `cost_noise` | `0.0` | observation noise during training episodes |

## Value convention

POMDPs.jl maximizes reward while DORA minimizes cost, so
`value(planner, s)` returns the *negated* learned cost-to-goal. States that
were collapsed into a sink during tabularization (absorbing reward cells)
return `0.0` from `value` and fall back to the first action from `action`.

## The episode-level API

For full control over the learning loop — as used by the paper's
experiments — bypass the solver interface and drive the learner directly:

```julia
tab = tabularize(mdp; start, classify, cost, c_min, c_to, c_crash,
                 horizon, cost_noise)
L = DORALearner(tab, K)
for k in 1:K
    pi = plan!(L)                 # policy vector over tabular indices
    run_episode!(L, pi, rng)      # simulate and feed observations back
end
```

The baseline learners from the paper (`DORA0`, `CED`, `EGD`, `OptimisticVI`,
`Sarsa`, `MCTSPlan`) and the chance-constrained `RiskDORA` (with
`update_multiplier!`) share the same API. The package also ships the
warehouse navigation benchmark domain ([`build`](@ref)), which implements
the POMDPs.jl MDP interface.
