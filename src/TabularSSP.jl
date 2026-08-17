"""
Generic tabular stochastic shortest path models built from POMDPs.jl problems.

`tabularize` enumerates the states of a discrete POMDPs.jl model that are
reachable from a designated start state, converts the reward structure into
positive traversal costs, and collapses every terminal state into a single
goal index and a single crash index. The result exposes the same arrays as the
warehouse model in `NavSSP.jl`, so the learners in `Learners.jl` run on it
without modification.

The determinized graph is obtained by mapping each state action pair to its
most likely successor. The self outcome and the crash outcome never define an
edge, which matches the convention used for the warehouse map, where a blocked
move has no determinized edge.
"""
module TabularSSPs

using POMDPs
using POMDPTools: weighted_iterator

export TabularSSP, tabularize

struct TabularSSP
    NS::Int                      # number of states including goal and crash
    NA::Int
    MAXOUT::Int
    S::Int                       # number of ordinary states, 1:S
    crash::Int
    goal::Int
    start::Int
    horizon::Int
    c_min::Float64
    c_to::Float64
    c_crash::Float64
    c_bump::Float64              # kept at zero, present for interface parity
    cost_noise::Float64
    succ::Matrix{Int}            # determinized successor, 0 when absent
    avail::BitMatrix
    out_s::Array{Int,3}
    out_p::Array{Float64,3}
    out_bump::Array{Float64,3}   # kept at zero, present for interface parity
    cout::Array{Float64,3}       # traversal cost of each outcome
    cbar::Matrix{Float64}        # expected traversal cost
    name::String
    states::Vector{Any}          # original model state of each ordinary index
end

"""
    tabularize(mdp; start, classify, cost, c_min, c_to, c_crash, horizon,
               cost_noise, key=identity, name="tabular")

Build a `TabularSSP` from a POMDPs.jl model with discrete states and actions.

`start` is the initial state of the model. `classify(sp)` returns `:goal`,
`:crash`, or `:normal` for a successor state. `cost(s, a, sp)` returns the
traversal cost of the outcome and must be at least `c_min`. `key(s)` maps a
state to a hashable identifier and is needed when the state type does not
hash by value.
"""
function tabularize(mdp; start, classify, cost, c_min::Float64,
                    c_to::Float64, c_crash::Float64, horizon::Int,
                    cost_noise::Float64, key=identity, name::String="tabular")
    A = collect(actions(mdp))
    NA = length(A)

    idx = Dict{Any,Int}(key(start) => 1)
    slist = Any[start]
    # per (state index, action index): outcome key => (prob, prob weighted cost)
    raw = Vector{Vector{Dict{Any,Tuple{Float64,Float64}}}}()

    head = 1
    while head <= length(slist)
        s = slist[head]
        row = [Dict{Any,Tuple{Float64,Float64}}() for _ in 1:NA]
        for (ai, a) in enumerate(A)
            for (sp, p) in weighted_iterator(transition(mdp, s, a))
                p > 0.0 || continue
                cls = classify(sp)
                c = cost(s, a, sp)
                if cls == :normal
                    kk = key(sp)
                    if !haskey(idx, kk)
                        idx[kk] = length(slist) + 1
                        push!(slist, sp)
                    end
                    tk = idx[kk]
                else
                    tk = cls
                end
                pc = get(row[ai], tk, (0.0, 0.0))
                row[ai][tk] = (pc[1] + p, pc[2] + p * c)
            end
        end
        push!(raw, row)
        head += 1
    end

    S = length(slist)
    NS = S + 2
    goal = S + 1
    crash = S + 2
    MAXOUT = max(1, maximum(length(row[ai]) for row in raw for ai in 1:NA))

    succ = zeros(Int, NS, NA)
    avail = falses(NS, NA)
    out_s = zeros(Int, NS, NA, MAXOUT)
    out_p = zeros(NS, NA, MAXOUT)
    out_bump = zeros(NS, NA, MAXOUT)
    cout = zeros(NS, NA, MAXOUT)
    cbar = zeros(NS, NA)

    for s in 1:S, ai in 1:NA
        entries = [(tk === :goal ? goal : tk === :crash ? crash : tk::Int,
                    p, pcsum) for (tk, (p, pcsum)) in raw[s][ai]]
        sort!(entries, by=first)     # deterministic outcome order
        best_p = 0.0
        for (k, (t, p, pcsum)) in enumerate(entries)
            out_s[s, ai, k] = t
            out_p[s, ai, k] = p
            cout[s, ai, k] = pcsum / p
            cbar[s, ai] += pcsum
            if t != crash && t != s && p > best_p
                best_p = p
                succ[s, ai] = t
            end
        end
        avail[s, ai] = succ[s, ai] > 0
    end

    for a in 1:NA
        out_s[goal, a, 1] = goal
        out_p[goal, a, 1] = 1.0
        out_s[crash, a, 1] = crash
        out_p[crash, a, 1] = 1.0
        avail[crash, a] = true
    end

    return TabularSSP(NS, NA, MAXOUT, S, crash, goal, 1, horizon, c_min, c_to,
                      c_crash, 0.0, cost_noise, succ, avail, out_s, out_p,
                      out_bump, cout, cbar, name, slist)
end

# ---------------- exact evaluation ----------------
# These mirror the routines in NavSSP.jl with the sizes taken from the model.

function eval_policy(m::TabularSSP, pi::Vector{Int})
    V = fill(m.c_to, m.NS)
    V[m.goal] = 0.0
    V[m.crash] = m.c_crash
    Vn = similar(V)
    for _ in 1:m.horizon
        @inbounds for s in 1:m.NS
            a = pi[s]
            acc = m.cbar[s, a]
            for k in 1:m.MAXOUT
                p = m.out_p[s, a, k]
                p > 0.0 && (acc += p * V[m.out_s[s, a, k]])
            end
            Vn[s] = acc
        end
        Vn[m.goal] = 0.0
        Vn[m.crash] = m.c_crash
        copyto!(V, Vn)
    end
    return V
end

function optimal_value(m::TabularSSP)
    V = fill(m.c_to, m.NS)
    V[m.goal] = 0.0
    V[m.crash] = m.c_crash
    Vn = similar(V)
    for _ in 1:m.horizon
        @inbounds for s in 1:m.NS
            best = Inf
            for a in 1:m.NA
                acc = m.cbar[s, a]
                for k in 1:m.MAXOUT
                    p = m.out_p[s, a, k]
                    p > 0.0 && (acc += p * V[m.out_s[s, a, k]])
                end
                acc < best && (best = acc)
            end
            Vn[s] = best
        end
        Vn[m.goal] = 0.0
        Vn[m.crash] = m.c_crash
        copyto!(V, Vn)
    end
    pi = ones(Int, m.NS)
    @inbounds for s in 1:m.NS
        best, ba = Inf, 1
        for a in 1:m.NA
            acc = m.cbar[s, a]
            for k in 1:m.MAXOUT
                p = m.out_p[s, a, k]
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
    return V, pi
end

function outcome_rates(m::TabularSSP, pi::Vector{Int})
    f = zeros(m.NS)
    g = zeros(m.NS)
    f[m.goal] = 1.0
    g[m.crash] = 1.0
    fn, gn = similar(f), similar(g)
    for _ in 1:m.horizon
        @inbounds for s in 1:m.NS
            a = pi[s]
            af, ag = 0.0, 0.0
            for k in 1:m.MAXOUT
                p = m.out_p[s, a, k]
                if p > 0.0
                    z = m.out_s[s, a, k]
                    af += p * f[z]
                    ag += p * g[z]
                end
            end
            fn[s], gn[s] = af, ag
        end
        fn[m.goal], gn[m.goal] = 1.0, 0.0
        fn[m.crash], gn[m.crash] = 0.0, 1.0
        copyto!(f, fn)
        copyto!(g, gn)
    end
    sr, cr = f[m.start], g[m.start]
    return sr, cr, max(0.0, 1.0 - sr - cr)
end

function causality_margin(m::TabularSSP, V::Vector{Float64}, pi::Vector{Int})
    worst = Inf
    okc, tot = 0, 0
    for s in 1:m.S
        s == m.goal && continue
        a = pi[s]
        good = true
        for k in 1:m.MAXOUT
            p = m.out_p[s, a, k]
            (p <= 0.0) && continue
            z = m.out_s[s, a, k]
            (z == m.crash || z == s) && continue
            mg = V[s] - V[z]
            worst = min(worst, mg)
            mg <= 0.0 && (good = false)
        end
        tot += 1
        good && (okc += 1)
    end
    return worst, okc / max(tot, 1)
end

function reduced_costs(m::TabularSSP, V::Vector{Float64})
    w = fill(Inf, m.NS, m.NA)
    @inbounds for s in 1:m.NS, a in 1:m.NA
        sp = m.succ[s, a]
        sp == 0 && continue
        q = m.cbar[s, a]
        for k in 1:m.MAXOUT
            p = m.out_p[s, a, k]
            p > 0.0 && (q += p * V[m.out_s[s, a, k]])
        end
        w[s, a] = q - V[sp]
    end
    return w
end

end # module
