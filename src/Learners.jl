"""
Online learners for the navigation SSP.

DORA is the proposed method. It keeps optimistic estimates of the SA traversal
costs and calls Dijkstra a fixed number of times per episode on the known
determinized map. The weight handed to the oracle is a reduced cost: the
learned step cost plus the expected change in cost-to-goal caused by actuation
slip. At the fixed point the weight equals Q(s,a) - d(sigma(s,a)), so the
shortest path tree reproduces the greedy policy of the stochastic model.
"""
module Learners

using ..RNGs
using ..NavSSPs
using ..TabularSSPs
using ..Dijkstra

export DORALearner, DORA0, CED, EGD, OptimisticVI, RiskDORA, Sarsa, MCTSPlan,
       plan!, observe!, run_episode!, update_multiplier!, work, oracle_calls,
       vi_sweeps, rhat

abstract type Learner end

mutable struct Counters
    plan_ns::Int
    oracle::Int
    sweeps::Int
    work::Int
end
Counters() = Counters(0, 0, 0, 0)

work(L::Learner) = L.ct.work
oracle_calls(L::Learner) = L.ct.oracle
vi_sweeps(L::Learner) = L.ct.sweeps

"Number of actions and maximum number of outcomes of a model."
nact(m) = size(m.cbar, 2)
maxout(m) = size(m.out_p, 3)

# ---------------- shared statistics ----------------

mutable struct CostStats
    N::Matrix{Float64}
    Csum::Matrix{Float64}
    beta::Float64
    delta::Float64
    K::Int
end
CostStats(ns, na, K; beta=0.05, delta=0.1) =
    CostStats(zeros(ns, na), zeros(ns, na), beta, delta, K)

function chat(st::CostStats, c_min::Float64)
    c = st.Csum ./ max.(st.N, 1.0)
    @inbounds for i in eachindex(c)
        st.N[i] == 0.0 && (c[i] = c_min)
    end
    return c
end

function radius(st::CostStats, ns::Int, na::Int)
    L = log(2.0 * ns * na * max(st.K, 2) / st.delta)
    return st.beta .* sqrt.(L ./ max.(st.N, 1.0))
end

function base_weights(st::CostStats, m, optimistic::Bool)
    c = chat(st, m.c_min)
    w = optimistic ? c .- radius(st, m.NS, nact(m)) : c
    return max.(m.c_min, w)
end

# ---------------- DORA ----------------

"""
Dijkstra Oracle Reduced cost Algorithm.

`iters` is the number of oracle calls per episode and `alpha` the damping of
the cost-to-goal update. Setting `correct = false` recovers plain determinize
and replan.
"""
mutable struct DORALearner{M} <: Learner
    m::M
    st::CostStats
    ct::Counters
    rev::ReverseAdj
    d::Vector{Float64}
    iters::Int
    alpha::Float64
    optimistic::Bool
    correct::Bool
    explore_eps::Float64
    wextra::Matrix{Float64}      # additive weight term, used by DORA-S
    name::String
end

function DORALearner(m, K::Int; iters::Int=3, alpha::Float64=0.4,
              optimistic::Bool=true, correct::Bool=true, beta::Float64=0.05,
              explore_eps::Float64=0.0, name::String="DORA")
    d = zeros(m.NS)
    d[m.crash] = m.c_crash
    return DORALearner(m, CostStats(m.NS, nact(m), K; beta=beta), Counters(),
                ReverseAdj(m.succ), d, iters, alpha, optimistic, correct,
                explore_eps, zeros(m.NS, nact(m)), name)
end

DORA0(m, K::Int; kw...) =
    DORALearner(m, K; iters=1, correct=false, optimistic=true, name="DORA-0", kw...)
CED(m, K::Int; kw...) =
    DORALearner(m, K; iters=1, correct=false, optimistic=false, name="CED", kw...)
EGD(m, K::Int; kw...) =
    DORALearner(m, K; iters=1, correct=false, optimistic=false, explore_eps=0.10,
         name="EGD", kw...)

"Extra expected cost of a step relative to arriving at the intended successor."
function drift!(out::Matrix{Float64}, m, d::Vector{Float64})
    @inbounds for s in 1:m.NS, a in 1:nact(m)
        acc = 0.0
        for k in 1:maxout(m)
            p = m.out_p[s, a, k]
            p > 0.0 && (acc += p * d[m.out_s[s, a, k]])
        end
        sp = m.succ[s, a]
        out[s, a] = sp == 0 ? 0.0 : acc - d[sp]
    end
    return out
end

function plan!(L::DORALearner)
    m = L.m
    t0 = time_ns()
    wb = base_weights(L.st, m, L.optimistic) .+ L.wextra
    w = similar(wb)
    dr = similar(wb)
    pi = ones(Int, m.NS)
    d = L.d
    for _ in 1:(L.correct ? L.iters : 1)
        if L.correct
            drift!(dr, m, d)
            L.ct.work += m.NS * nact(m) * maxout(m)
            @inbounds for i in eachindex(w)
                w[i] = max(0.0, wb[i] + dr[i])
            end
        else
            copyto!(w, wb)
        end
        @inbounds for s in 1:m.NS, a in 1:nact(m)
            (m.avail[s, a] && m.succ[s, a] > 0) || (w[s, a] = 1e6)
        end
        r0 = edge_scans()
        pi, dn = dijkstra_policy(L.rev, w, m.goal, m.avail, m.NS, nact(m))
        L.ct.work += edge_scans() - r0
        L.ct.oracle += 1
        @inbounds for s in 1:m.NS
            isfinite(dn[s]) || (dn[s] = m.c_to)
        end
        dn[m.crash] = m.c_crash
        if L.correct
            @inbounds for s in 1:m.NS
                d[s] = (1.0 - L.alpha) * d[s] + L.alpha * dn[s]
            end
        else
            copyto!(d, dn)
        end
        d[m.crash] = m.c_crash
    end
    L.ct.plan_ns += Int(time_ns() - t0)
    return pi
end

# ---------------- optimistic value iteration ----------------

mutable struct OptimisticVI{M} <: Learner
    m::M
    st::CostStats
    ct::Counters
    V::Vector{Float64}
    Ncount::Array{Float64,3}
    known_P::Bool
    sweeps::Int
    tol::Float64
    explore_eps::Float64
    name::String
end

function OptimisticVI(m, K::Int; known_P::Bool=false, sweeps::Int=200,
                      tol::Float64=1e-4, beta::Float64=0.05)
    V = zeros(m.NS)
    V[m.crash] = m.c_crash
    return OptimisticVI(m, CostStats(m.NS, nact(m), K; beta=beta), Counters(), V,
                        zeros(m.NS, nact(m), maxout(m)), known_P, sweeps, tol, 0.0,
                        known_P ? "OVI-K" : "OVI-U")
end

function phat(L::OptimisticVI)
    m = L.m
    P = similar(m.out_p)
    @inbounds for s in 1:m.NS, a in 1:nact(m)
        tot = 0.0
        for k in 1:maxout(m)
            tot += L.Ncount[s, a, k]
        end
        for k in 1:maxout(m)
            P[s, a, k] = tot > 0.0 ? L.Ncount[s, a, k] / tot : m.out_p[s, a, k]
        end
    end
    return P
end

function plan!(L::OptimisticVI)
    m = L.m
    t0 = time_ns()
    w = base_weights(L.st, m, true)
    @inbounds for s in 1:m.NS, a in 1:nact(m)
        m.avail[s, a] || (w[s, a] = 1e6)
    end
    P = L.known_P ? m.out_p : phat(L)
    V = L.V
    Vn = similar(V)
    done = 0
    for _ in 1:L.sweeps
        gap = 0.0
        @inbounds for s in 1:m.NS
            best = Inf
            for a in 1:nact(m)
                acc = w[s, a]
                for k in 1:maxout(m)
                    p = P[s, a, k]
                    p > 0.0 && (acc += p * V[m.out_s[s, a, k]])
                end
                acc < best && (best = acc)
            end
            Vn[s] = min(best, m.c_to)
        end
        Vn[m.goal] = 0.0
        Vn[m.crash] = m.c_crash
        @inbounds for s in 1:m.NS
            gap = max(gap, abs(Vn[s] - V[s]))
        end
        copyto!(V, Vn)
        done += 1
        gap < L.tol && break
    end
    pi = ones(Int, m.NS)
    @inbounds for s in 1:m.NS
        best, ba = Inf, 1
        for a in 1:nact(m)
            acc = w[s, a]
            for k in 1:maxout(m)
                p = P[s, a, k]
                p > 0.0 && (acc += p * V[m.out_s[s, a, k]])
            end
            if acc < best
                best, ba = acc, a
            end
        end
        pi[s] = ba
    end
    pi[m.goal] = 1
    pi[m.crash] = 1
    L.ct.plan_ns += Int(time_ns() - t0)
    L.ct.sweeps += done
    L.ct.work += (done + 1) * m.NS * nact(m) * maxout(m)
    return pi
end

# ---------------- chance constrained DORA ----------------

"""
DORA with an explicit risk weight. The additive term `-log(1 - phat)` makes the
weight of a route equal to the negative log probability of traversing it
without contact, so a shortest path under the combined weight trades distance
against survival. The multiplier is updated by projected dual ascent on the
realized contact indicator, which needs no extra planning sweep.
"""
mutable struct RiskDORA <: Learner
    inner::DORALearner
    Rsum::Matrix{Float64}
    budget::Float64
    lam::Float64
    lam_max::Float64
    eta0::Float64
    step::Int
    name::String
end

function RiskDORA(m, K::Int; budget::Float64=0.05, lam0::Float64=0.0,
                  lam_max::Float64=300.0, eta0::Float64=300.0, kw...)
    return RiskDORA(DORALearner(m, K; kw...), zeros(m.NS, nact(m)), budget, lam0,
                    lam_max, eta0, 0, "DORA-S")
end

work(L::RiskDORA) = L.inner.ct.work
oracle_calls(L::RiskDORA) = L.inner.ct.oracle
vi_sweeps(::RiskDORA) = 0

"Shrinkage estimate of the contact probability. Unvisited pairs are pulled to
zero so that cost optimism, not risk pessimism, drives exploration."
rhat(L::RiskDORA) = clamp.(L.Rsum ./ (L.inner.st.N .+ 1.0), 0.0, 0.95)

function plan!(L::RiskDORA)
    L.inner.wextra .= L.lam .* (-1.0 .* log.(1.0 .- rhat(L)))
    return plan!(L.inner)
end

"Projected dual ascent driven by the realized contact indicator."
function update_multiplier!(L::RiskDORA, contact::Bool)
    L.step += 1
    eta = L.eta0 / sqrt(L.step)
    g = (contact ? 1.0 : 0.0) - L.budget
    L.lam = clamp(L.lam + eta * g, 0.0, L.lam_max)
    return L.lam
end

# ---------------- expected SARSA ----------------

"""
Tabular expected SARSA with an epsilon greedy behavior policy. The method is
model free. It never plans, so its work per episode is one temporal difference
update per step plus one greedy sweep to extract the policy.
"""
mutable struct Sarsa{M} <: Learner
    m::M
    st::CostStats
    ct::Counters
    Q::Matrix{Float64}
    explore_eps::Float64
    name::String
end

function Sarsa(m, K::Int; explore_eps::Float64=0.10, beta::Float64=0.05)
    return Sarsa(m, CostStats(m.NS, nact(m), K; beta=beta), Counters(),
                 zeros(m.NS, nact(m)), explore_eps, "SARSA")
end

function plan!(L::Sarsa)
    m = L.m
    pi = ones(Int, m.NS)
    @inbounds for s in 1:m.NS
        best, ba = Inf, 1
        for a in 1:nact(m)
            m.avail[s, a] || continue
            L.Q[s, a] < best && ((best, ba) = (L.Q[s, a], a))
        end
        pi[s] = ba
    end
    pi[m.goal] = 1
    pi[m.crash] = 1
    L.ct.work += m.NS * nact(m)
    return pi
end

"Expected SARSA update under the epsilon greedy behavior policy."
function sarsa_step!(L::Sarsa, s::Int, a::Int, cost::Float64, sp::Int)
    m = L.m
    if sp == m.goal
        tail = 0.0
    elseif sp == m.crash
        tail = m.c_crash
    else
        best, tot, na = Inf, 0.0, 0
        @inbounds for ap in 1:nact(m)
            m.avail[sp, ap] || continue
            q = L.Q[sp, ap]
            q < best && (best = q)
            tot += q
            na += 1
        end
        na == 0 && (best = m.c_to; tot = m.c_to; na = 1)
        tail = (1.0 - L.explore_eps) * best + L.explore_eps * tot / na
    end
    n = L.st.N[s, a]
    alpha = min(0.5, 10.0 / (10.0 + n))
    L.Q[s, a] += alpha * (cost + tail - L.Q[s, a])
    L.ct.work += nact(m)
    return nothing
end

# ---------------- MCTS with the true generative model ----------------

"""
UCT planner that is given the true generative model, including the expected
step costs. Nothing is learned, so the extracted policy is stationary and the
search runs once. The reported work is the total number of generative model
steps divided by the number of episodes, which is the amortized cost of the
search.
"""
mutable struct MCTSPlan{M} <: Learner
    m::M
    st::CostStats
    ct::Counters
    pi::Vector{Int}
    sims::Int
    depth::Int
    cucb::Float64
    rng::SplitMix64
    planned::Bool
    name::String
end

function MCTSPlan(m, K::Int; sims::Int=1000, depth::Int=60,
                  cucb::Float64=60.0, seed::Int=1234, beta::Float64=0.05)
    return MCTSPlan(m, CostStats(m.NS, nact(m), K; beta=beta), Counters(),
                    ones(Int, m.NS), sims, depth, cucb, SplitMix64(seed),
                    false, "MCTS")
end

"Sample one generative model step. Returns the successor and the step cost."
function model_step(L::MCTSPlan, s::Int, a::Int)
    m = L.m
    u = rand01(L.rng)
    acc = 0.0
    k = 1
    @inbounds for kk in 1:maxout(m)
        acc += m.out_p[s, a, kk]
        if u <= acc
            k = kk
            break
        end
    end
    L.ct.work += 1
    return m.out_s[s, a, k], m.cbar[s, a]
end

"Uniform random rollout to the depth cap."
function mcts_rollout(L::MCTSPlan, s::Int, depth::Int)
    m = L.m
    total = 0.0
    @inbounds for _ in depth:L.depth
        s == m.goal && return total
        s == m.crash && return total + m.c_crash
        a = randint(L.rng, nact(m)) + 1
        m.avail[s, a] || continue
        s, c = model_step(L, s, a)
        total += c
    end
    s == m.goal && return total
    s == m.crash && return total + m.c_crash
    return total + m.c_to
end

function mcts_sim!(L::MCTSPlan, s::Int, depth::Int, Ns::Dict{Int,Int},
                   Nsa::Dict{Tuple{Int,Int},Int},
                   Qsa::Dict{Tuple{Int,Int},Float64})
    m = L.m
    s == m.goal && return 0.0
    s == m.crash && return m.c_crash
    depth >= L.depth && return m.c_to
    if !haskey(Ns, s)
        Ns[s] = 0
        return mcts_rollout(L, s, depth)
    end
    logN = log(Float64(max(Ns[s], 1)))
    best_a, best_v = 0, Inf
    @inbounds for a in 1:nact(m)
        m.avail[s, a] || continue
        n = get(Nsa, (s, a), 0)
        if n == 0
            best_a = a
            break
        end
        v = Qsa[(s, a)] - L.cucb * sqrt(logN / n)
        v < best_v && ((best_v, best_a) = (v, a))
    end
    best_a == 0 && return m.c_to
    sp, c = model_step(L, s, best_a)
    G = c + mcts_sim!(L, sp, depth + 1, Ns, Nsa, Qsa)
    Ns[s] += 1
    n = get(Nsa, (s, best_a), 0) + 1
    Nsa[(s, best_a)] = n
    q = get(Qsa, (s, best_a), 0.0)
    Qsa[(s, best_a)] = q + (G - q) / n
    return G
end

function plan!(L::MCTSPlan)
    L.planned && return L.pi
    m = L.m
    t0 = time_ns()
    @inbounds for s in 1:m.NS
        (s == m.goal || s == m.crash) && (L.pi[s] = 1; continue)
        Ns = Dict{Int,Int}()
        Nsa = Dict{Tuple{Int,Int},Int}()
        Qsa = Dict{Tuple{Int,Int},Float64}()
        for _ in 1:L.sims
            mcts_sim!(L, s, 0, Ns, Nsa, Qsa)
        end
        best, ba = Inf, 1
        for a in 1:nact(m)
            m.avail[s, a] || continue
            n = get(Nsa, (s, a), 0)
            n == 0 && continue
            Qsa[(s, a)] < best && ((best, ba) = (Qsa[(s, a)], a))
        end
        L.pi[s] = ba
    end
    L.ct.plan_ns += Int(time_ns() - t0)
    L.ct.oracle += 1
    L.planned = true
    return L.pi
end

# ---------------- statistics updates and simulation ----------------

stats(L::DORALearner) = L.st
stats(L::OptimisticVI) = L.st
stats(L::RiskDORA) = L.inner.st
stats(L::Sarsa) = L.st
stats(L::MCTSPlan) = L.st
model(L::DORALearner) = L.m
model(L::OptimisticVI) = L.m
model(L::RiskDORA) = L.inner.m
model(L::Sarsa) = L.m
model(L::MCTSPlan) = L.m
explore(L::DORALearner) = L.explore_eps
explore(L::OptimisticVI) = L.explore_eps
explore(::RiskDORA) = 0.0
explore(L::Sarsa) = L.explore_eps
explore(::MCTSPlan) = 0.0

function observe!(L::Learner, s::Int, a::Int, cost::Float64)
    st = stats(L)
    st.N[s, a] += 1.0
    st.Csum[s, a] += cost
    return nothing
end

"""
    run_episode!(L, pi, rng)

Simulate one episode and feed the observations back to the learner. Returns
`(cost, success, collided, timeout)`.
"""
run_episode!(L::Learner, pi::Vector{Int}, rng::SplitMix64) =
    simulate_episode!(model(L), L, pi, rng)

"Warehouse episode. The observed step cost is the terrain sample plus a bump."
function simulate_episode!(m::NavSSP, L::Learner, pi::Vector{Int},
                           rng::SplitMix64)
    s = m.start
    total = 0.0
    e = explore(L)
    pv = zeros(MAXOUT)
    for _ in 1:m.horizon
        s == m.goal && return (total, true, false, false)
        a = pi[s]
        if e > 0.0 && rand01(rng) < e
            a = randint(rng, NACT) + 1
        end
        r0, c0 = m.cells[s]
        step = max(m.c_min,
                   m.terrain[r0+1, c0+1] + uniform(rng, -m.cost_noise, m.cost_noise))
        @inbounds for k in 1:MAXOUT
            pv[k] = m.out_p[s, a, k]
        end
        k = categorical(rng, pv)
        sp = m.out_s[s, a, k]
        m.out_bump[s, a, k] > 0.5 && (step += m.c_bump)
        crashed = (sp == m.crash)
        observe!(L, s, a, step)
        if L isa OptimisticVI
            L.Ncount[s, a, k] += 1.0
        elseif L isa RiskDORA
            crashed && (L.Rsum[s, a] += 1.0)
        elseif L isa Sarsa
            sarsa_step!(L, s, a, step, sp)
        end
        total += step
        s = sp
        crashed && return (total + m.c_crash, false, true, false)
    end
    s == m.goal && return (total, true, false, false)
    return (total + m.c_to, false, false, true)
end

"Tabular episode. The observed step cost is the outcome cost plus noise."
function simulate_episode!(m::TabularSSP, L::Learner, pi::Vector{Int},
                           rng::SplitMix64)
    s = m.start
    total = 0.0
    e = explore(L)
    pv = zeros(m.MAXOUT)
    for _ in 1:m.horizon
        s == m.goal && return (total, true, false, false)
        a = pi[s]
        if e > 0.0 && rand01(rng) < e
            a = randint(rng, m.NA) + 1
        end
        u = uniform(rng, -m.cost_noise, m.cost_noise)
        @inbounds for k in 1:m.MAXOUT
            pv[k] = m.out_p[s, a, k]
        end
        k = categorical(rng, pv)
        sp = m.out_s[s, a, k]
        step = max(m.c_min, m.cout[s, a, k] + u)
        crashed = (sp == m.crash)
        observe!(L, s, a, step)
        if L isa OptimisticVI
            L.Ncount[s, a, k] += 1.0
        elseif L isa RiskDORA
            crashed && (L.Rsum[s, a] += 1.0)
        elseif L isa Sarsa
            sarsa_step!(L, s, a, step, sp)
        end
        total += step
        s = sp
        crashed && return (total + m.c_crash, false, true, false)
    end
    s == m.goal && return (total, true, false, false)
    return (total + m.c_to, false, false, true)
end

end # module
