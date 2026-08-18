import POMDPs
import POMDPs: MDP, Solver, Policy, actions, transition, isterminal,
               initialstate, discount
using POMDPTools: weighted_iterator
import Random

"""
    DORASolver(; kwargs...)

POMDPs.jl solver for the Dijkstra Oracle Reduced-cost Algorithm. `solve`
tabularizes a discrete MDP into a stochastic shortest path problem and returns
a [`DORAPlanner`](@ref) that plans at decision time with a fixed number of
reduced-cost Dijkstra calls, in the style of other online solvers such as
MCTS.jl.

Learner hyperparameters (defaults are the paper values):

- `iters = 3`: Dijkstra oracle calls per (re)planning round
- `alpha = 0.4`: damping of the cost-to-goal update
- `beta = 0.05`: confidence radius scale for optimistic cost estimates
- `optimistic = true`, `correct = true`, `explore_eps = 0.0`: see `DORALearner`
- `known_costs = true`: seed the cost statistics from the model's expected
  step costs, so the first `action` call already yields the reduced-cost
  Dijkstra policy with no environment feedback. Set to `false` to learn the
  costs online through `observe!`.
- `train_episodes = 0`: number of self-simulated training episodes to run
  inside `solve` (the protocol of experiment 4 in the paper). The model's
  transition kernel is used as the generative simulator.
- `episodes_budget = 500`: the horizon `K` in the confidence radius schedule
- `seed = 0`: seed of the internal `SplitMix64` stream

Tabularization controls (`nothing` means "derive a default from the MDP"):

- `start`: initial state; default samples `initialstate(mdp)`
- `classify`: function `sp -> :goal | :crash | :normal`. The default classifies
  a *terminal* state as `:crash` when its reward is negative and `:goal`
  otherwise, and a *non-terminal* state as `:goal` for positive reward,
  `:crash` for negative reward, and `:normal` for zero. The sign is taken from
  the largest expected immediate reward over actions.
- `cost`: function `(s, a, sp) -> Float64`; default `max(c_min,
  step_cost - reward(mdp, s, a))`
- `key = identity`: hashable identifier of a state (needed when the state type
  does not hash by value)
- `step_cost = 1.0`, `c_min = 0.25`, `horizon = 200`, `cost_noise = 0.0`
- `c_to` (default `2 * horizon * step_cost`): timeout penalty
- `c_crash` (default `horizon * step_cost`): crash penalty
- `name = "tabular"`: label stored on the resulting `TabularSSP`, used only to
  identify the model in reports and plots

DORA treats the model as an undiscounted stochastic shortest path problem; a
warning is issued when `discount(mdp) < 1`.

!!! note "`optimistic` only takes effect when the costs are unknown"
    The learner is constructed with `optimistic && !known_costs`. The
    optimistic confidence radius is an exploration bonus for costs that still
    have to be discovered, so under the default `known_costs = true` the costs
    are already exact and the radius is suppressed — setting
    `optimistic = true` there changes nothing. Optimistic exploration requires
    `known_costs = false`.
"""
Base.@kwdef mutable struct DORASolver <: Solver
    iters::Int = 3
    alpha::Float64 = 0.4
    beta::Float64 = 0.05
    optimistic::Bool = true
    correct::Bool = true
    explore_eps::Float64 = 0.0
    known_costs::Bool = true
    train_episodes::Int = 0
    episodes_budget::Int = 500
    seed::Int = 0
    start::Any = nothing
    classify::Union{Nothing,Function} = nothing
    cost::Union{Nothing,Function} = nothing
    key::Function = identity
    step_cost::Float64 = 1.0
    c_min::Float64 = 0.25
    horizon::Int = 200
    c_to::Union{Nothing,Float64} = nothing
    c_crash::Union{Nothing,Float64} = nothing
    cost_noise::Float64 = 0.0
    name::String = "tabular"
end

"""
    DORAPlanner

Decision-time policy returned by `solve(::DORASolver, ::MDP)`. Planning is
lazy: the Dijkstra oracle calls run on the first `action` query and again
whenever new cost observations arrive through `observe!`.

`action(p, s)` returns the planned action for `s`. States that were collapsed
into the goal or crash sink during tabularization (absorbing reward cells) and
states unreachable from the start fall back to the first action of the model,
since no planning information exists for them.

`value(p, s)` returns the negated learned cost-to-goal, matching the POMDPs.jl
convention that values are maximized. States outside the tabularization return
`0.0`.
"""
mutable struct DORAPlanner{M<:MDP,A} <: Policy
    mdp::M
    tab::TabularSSP
    learner::DORALearner{TabularSSP}
    acts::Vector{A}
    sindex::Dict{Any,Int}
    key::Function
    pi::Vector{Int}
    dirty::Bool
    rng::SplitMix64
end

"Expected immediate reward of `(s, a)`, tolerating both reward arities."
function _sar(mdp::MDP, s, a)
    if applicable(POMDPs.reward, mdp, s, a)
        return POMDPs.reward(mdp, s, a)
    end
    d = transition(mdp, s, a)
    return sum(p * POMDPs.reward(mdp, s, a, sp)
               for (sp, p) in weighted_iterator(d); init=0.0)
end

function _default_classify(mdp::MDP, A)
    return sp -> begin
        r = maximum(_sar(mdp, sp, a) for a in A)
        if isterminal(mdp, sp)
            return r < 0.0 ? :crash : :goal
        end
        return r > 0.0 ? :goal : r < 0.0 ? :crash : :normal
    end
end

function _tabularize(sol::DORASolver, mdp::MDP)
    A = collect(actions(mdp))
    start = sol.start === nothing ?
        rand(Random.MersenneTwister(sol.seed), initialstate(mdp)) : sol.start
    classify = sol.classify === nothing ? _default_classify(mdp, A) : sol.classify
    cost = sol.cost === nothing ?
        ((s, a, sp) -> max(sol.c_min, sol.step_cost - _sar(mdp, s, a))) : sol.cost
    c_to = sol.c_to === nothing ? 2.0 * sol.horizon * sol.step_cost : sol.c_to
    c_crash = sol.c_crash === nothing ? sol.horizon * sol.step_cost : sol.c_crash
    tab = tabularize(mdp; start=start, classify=classify, cost=cost,
                     c_min=sol.c_min, c_to=c_to, c_crash=c_crash,
                     horizon=sol.horizon, cost_noise=sol.cost_noise,
                     key=sol.key, name=sol.name)
    return tab, A
end

function POMDPs.solve(sol::DORASolver, mdp::MDP)
    discount(mdp) < 1.0 &&
        @warn "DORA treats the model as an undiscounted SSP; discount is ignored" maxlog = 1
    tab, A = _tabularize(sol, mdp)
    K = max(sol.episodes_budget, sol.train_episodes, 1)
    L = DORALearner(tab, K; iters=sol.iters, alpha=sol.alpha, beta=sol.beta,
                    optimistic=sol.optimistic && !sol.known_costs,
                    correct=sol.correct, explore_eps=sol.explore_eps)
    if sol.known_costs
        L.st.N .= 1.0
        L.st.Csum .= tab.cbar
    end
    rng = SplitMix64(sol.seed)
    for _ in 1:sol.train_episodes
        run_episode!(L, plan!(L), rng)
    end
    sindex = Dict{Any,Int}(sol.key(s) => i for (i, s) in enumerate(tab.states))
    return DORAPlanner(mdp, tab, L, A, sindex, sol.key,
                       ones(Int, tab.NS), true, rng)
end

"""
    replan!(p::DORAPlanner)

Run the learner's Dijkstra oracle calls and refresh the cached policy.
Called automatically by `action` when observations have arrived since the
last plan.
"""
function replan!(p::DORAPlanner)
    p.pi = plan!(p.learner)
    p.dirty = false
    return p.pi
end

function POMDPs.action(p::DORAPlanner, s)
    p.dirty && replan!(p)
    i = get(p.sindex, p.key(s), 0)
    i == 0 && return p.acts[1]
    return p.acts[p.pi[i]]
end

function POMDPs.value(p::DORAPlanner, s)
    p.dirty && replan!(p)
    i = get(p.sindex, p.key(s), 0)
    i == 0 && return 0.0
    return -p.learner.d[i]
end

"""
    observe!(p::DORAPlanner, s, a, cost)

Report the observed traversal cost of taking action `a` in state `s`. Updates
the learner's cost statistics and marks the planner for lazy replanning, so
the Dijkstra oracle runs again on the next `action` query. This is the online
interface for deployment-time cost learning (`known_costs = false`).
"""
function Learners.observe!(p::DORAPlanner, s, a, cost::Real)
    i = get(p.sindex, p.key(s), 0)
    i == 0 && throw(ArgumentError("state $s is not part of the tabularized model"))
    ai = findfirst(isequal(a), p.acts)
    ai === nothing && throw(ArgumentError("action $a is not an action of the model"))
    observe!(p.learner, i, ai, Float64(cost))
    p.dirty = true
    return nothing
end

"""
    train!(p::DORAPlanner, episodes::Int)

Continue learning by running `episodes` self-simulated episodes on the
tabularized model (the protocol of experiment 4). Uses the planner's internal
`SplitMix64` stream.
"""
function train!(p::DORAPlanner, episodes::Int)
    for _ in 1:episodes
        run_episode!(p.learner, plan!(p.learner), p.rng)
    end
    p.dirty = true
    return p
end
