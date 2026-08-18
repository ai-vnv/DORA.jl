"""
Warehouse navigation as a stochastic shortest path problem, exposed through the
POMDPs.jl MDP interface.

The map geometry is known. Traversal costs and contact risk are unknown and
must be learned online. A commanded move succeeds with probability 1 - eps and
slips laterally otherwise. A slip into a shelf is a soft failure that stops the
robot in place. Entering a cell occupied by a dynamic obstacle is a hard
failure that ends the episode at the absorbing crash state.

States are indexed 1:NS where NS = S + 1 and index NS is the crash state.
POMDPs.jl maximizes reward, so `reward` returns the negative of the cost.
"""
module NavSSPs

using POMDPs
using POMDPTools: SparseCat, Deterministic
using ..RNGs

export NavSSP, build, eval_policy, optimal_value, outcome_rates,
       causality_margin, reduced_costs, MAXOUT, NACT

"""
    NACT

Number of actions of the warehouse domain: the four commanded moves N, E, S, W.
Exported because the arrays of a [`NavSSP`](@ref) are indexed with it; it is a
constant of this domain, not of the package (a [`TabularSSP`](@ref DORASolvers.TabularSSPs.TabularSSP) carries its
own `NA` field instead).
"""
const NACT = 4

"""
    MAXOUT

Number of stochastic outcome slots per state-action pair of the warehouse
domain: intended move, two lateral slips, and the crash sink. Exported for the
same reason as [`NACT`](@ref); a [`TabularSSP`](@ref DORASolvers.TabularSSPs.TabularSSP) carries its own `MAXOUT`
field, whose value depends on the model being tabularized.
"""
const MAXOUT = 4

const DR = (-1, 0, 1, 0)     # N, E, S, W
const DC = (0, 1, 0, -1)
const FREE, SHELF = 0, 1

# Validate the grid size against the fixed shelf layout. Internal: the
# user-facing contract lives in the `build` docstring.
#
# `build` places the start and the goal on the zero-based row `n - 12`, at
# columns `1` and `n - 2`. `warehouse_map` fills a row with shelves when its
# zero-based index is 2 or 3 modulo 4, and `n - 12 ≡ n (mod 4)`, so that row is
# a free aisle exactly when `n % 4` is 0 or 1. Any other size puts the start or
# the goal inside a shelf, which has no state index. That used to surface much
# later: as a BoundsError when the goal was blocked, and — when a cross aisle
# happened to spare the goal, as at n = 14 and n = 26 — as a silently corrupt
# model with `start == 0`. Sizes below 12 place the row above the top border.
function _check_grid_size(n::Int)
    n >= 12 || throw(ArgumentError(
        "NavSSP grid size n = $n is too small: the start and goal row is " *
        "n - 12 (zero based), so n must be at least 12."))
    r = n % 4
    (r == 0 || r == 1) || throw(ArgumentError(
        "NavSSP grid size n = $n places the start or the goal inside a " *
        "shelf row. The fixed warehouse layout requires n % 4 ∈ (0, 1), " *
        "for example 12, 13, 16, 17, 20, 21, 24, 25; got n % 4 = $r."))
    return nothing
end

# ---------------- map construction ----------------

function warehouse_map(n::Int)
    grid = zeros(Int, n, n)
    for r in 1:n, c in 1:n
        r0, c0 = r - 1, c - 1                      # zero based for the pattern
        in_shelf_row = (r0 % 4 == 2) || (r0 % 4 == 3)
        border = (r0 == 0) || (r0 == n - 1) || (c0 == 0) || (c0 == n - 1)
        cross_aisle = (c0 % 6 == 0)
        if in_shelf_row && !border && !cross_aisle
            grid[r, c] = SHELF
        end
    end
    return grid
end

function terrain_cost(grid::Matrix{Int}, c_min::Float64, rng, rough::Float64,
                      c_max::Float64=2.2)
    n = size(grid, 1)
    cost = zeros(n, n)
    for r in 1:n, c in 1:n
        near = 0
        for dr in -1:1, dc in -1:1
            rr, cc = r + dr, c + dc
            if 1 <= rr <= n && 1 <= cc <= n && grid[rr, cc] == SHELF
                near += 1
            end
        end
        cost[r, c] = 0.40 + 0.075 * near
    end
    if rough > 0.0 && rng !== nothing
        f = zeros(n, n)
        for r in 1:n, c in 1:n
            f[r, c] = rand01(rng)
        end
        for _ in 1:2
            g = zeros(n, n)
            for r in 1:n, c in 1:n
                acc, cnt = 0.0, 0
                for dr in -1:1, dc in -1:1
                    rr, cc = r + dr, c + dc
                    if 1 <= rr <= n && 1 <= cc <= n
                        acc += f[rr, cc]
                        cnt += 1
                    end
                end
                g[r, c] = acc / cnt
            end
            f = g
        end
        lo, hi = minimum(f), maximum(f)
        f = (f .- lo) ./ max(hi - lo, 1e-12)
        cost .+= rough .* f
    end
    return clamp.(cost, c_min, c_max)
end

function hazard_map(grid::Matrix{Int}, p_haz::Float64, band=(9, 10))
    n = size(grid, 1)
    haz = zeros(n, n)
    p_haz <= 0.0 && return haz
    for r in 2:(n-1), c0 in band
        c = c0 + 1
        if grid[r, c] == FREE
            haz[r, c] = p_haz
        end
    end
    return haz
end

# ---------------- the MDP ----------------

"""
    NavSSP

The warehouse navigation benchmark, exposed through the POMDPs.jl `MDP`
interface and constructed with [`build`](@ref). States are integer indices
`1:NS` where `NS = S + 1`, the ordinary cells are `1:S`, and index `NS` is the
absorbing crash state; `cells[s]` gives the zero-based `(row, col)` of cell `s`
and `sidx` is the inverse map.

The map geometry (`grid`, `succ`, `out_s`, `out_p`) is known to the planner,
while the traversal costs (`terrain`) and the contact risk (`haz`) are what a
learner has to discover online. `cbar[s, a]` is the expected step cost and
`pcrash[s, a]` the contact probability, both used only for exact evaluation.

The exported helpers `optimal_value`, `eval_policy`, `outcome_rates`,
`causality_margin` and `reduced_costs` accept a `NavSSP`, exactly as they
accept a [`TabularSSP`](@ref DORASolvers.TabularSSPs.TabularSSP).
"""
struct NavSSP <: MDP{Int,Int}
    n::Int
    eps::Float64
    c_bump::Float64
    c_crash::Float64
    c_to::Float64
    horizon::Int
    c_min::Float64
    cost_noise::Float64
    grid::Matrix{Int}
    terrain::Matrix{Float64}
    haz::Matrix{Float64}
    cells::Vector{Tuple{Int,Int}}
    sidx::Matrix{Int}
    S::Int
    NS::Int
    crash::Int
    start::Int
    goal::Int
    succ::Matrix{Int}
    avail::BitMatrix
    out_s::Array{Int,3}
    out_p::Array{Float64,3}
    out_bump::Array{Float64,3}
    cbar::Matrix{Float64}
    pcrash::Matrix{Float64}
end

"""
    build(; n=20, eps=0.10, c_bump=1.5, c_crash=80.0, c_to=60.0, horizon=150,
          c_min=0.25, p_haz=0.14, cost_noise=0.12, rough=0.0,
          terrain_rng=nothing) -> NavSSP

Construct the warehouse navigation SSP used as the paper's primary benchmark.
The defaults are the paper values.

`n` is the side length of the square grid. The shelf layout is fixed, and it
admits a start and a goal only for `n >= 12` with `n % 4 ∈ (0, 1)` — for
example `12`, `13`, `16`, `17`, `20`, `21`, `24`. Any other size throws an
`ArgumentError`.

`rough` adds a smoothed random field to the terrain costs. Generating it needs
a stream, so passing `rough > 0` without a `terrain_rng` throws an
`ArgumentError` rather than silently producing flat terrain. `terrain_rng`
should be a [`SplitMix64`](@ref) so that the generated terrain matches the
Python reference implementation bit for bit.
"""
function build(; n::Int=20, eps::Float64=0.10, c_bump::Float64=1.5,
               c_crash::Float64=80.0, c_to::Float64=60.0, horizon::Int=150,
               c_min::Float64=0.25, p_haz::Float64=0.14,
               cost_noise::Float64=0.12, rough::Float64=0.0,
               terrain_rng=nothing)
    _check_grid_size(n)
    (rough > 0.0 && terrain_rng === nothing) && throw(ArgumentError(
        "build received rough = $rough but no terrain_rng. Terrain roughness " *
        "needs a random stream; pass terrain_rng = SplitMix64(seed), or set " *
        "rough = 0.0 for flat terrain."))
    grid = warehouse_map(n)
    terrain = terrain_cost(grid, c_min, terrain_rng, rough)
    haz = hazard_map(grid, p_haz)

    cells = Tuple{Int,Int}[]
    for r0 in 0:(n-1), c0 in 0:(n-1)
        grid[r0+1, c0+1] == FREE && push!(cells, (r0, c0))
    end
    sidx = fill(0, n, n)
    for (i, (r0, c0)) in enumerate(cells)
        sidx[r0+1, c0+1] = i
    end
    S = length(cells)
    NS = S + 1
    crash = NS
    start = sidx[n-12+1, 2]
    goal = sidx[n-12+1, n-1]

    succ = zeros(Int, NS, NACT)
    avail = falses(NS, NACT)
    out_s = zeros(Int, NS, NACT, MAXOUT)
    out_p = zeros(NS, NACT, MAXOUT)
    out_bump = zeros(NS, NACT, MAXOUT)
    cbar = zeros(NS, NACT)
    pcrash = zeros(NS, NACT)

    for (s, (r0, c0)) in enumerate(cells), a in 1:NACT
        rr, cc = r0 + DR[a], c0 + DC[a]
        if 0 <= rr < n && 0 <= cc < n && grid[rr+1, cc+1] == FREE
            avail[s, a] = true
            succ[s, a] = sidx[rr+1, cc+1]
        end
    end

    for (s, (r0, c0)) in enumerate(cells), a in 1:NACT
        if !avail[s, a]
            out_s[s, a, 1] = s
            out_p[s, a, 1] = 1.0
            out_bump[s, a, 1] = 1.0
            cbar[s, a] = terrain[r0+1, c0+1] + c_bump
            continue
        end
        ks = (a, mod1(a + 1, NACT), mod1(a + 3, NACT))
        base = (1.0 - eps, eps / 2, eps / 2)
        pc, pb = 0.0, 0.0
        for k in 1:3
            b = ks[k]
            rr, cc = r0 + DR[b], c0 + DC[b]
            ok = 0 <= rr < n && 0 <= cc < n && grid[rr+1, cc+1] == FREE
            if ok
                z = sidx[rr+1, cc+1]
                h = haz[rr+1, cc+1]
                out_s[s, a, k] = z
                out_p[s, a, k] = base[k] * (1.0 - h)
                pc += base[k] * h
            else
                out_s[s, a, k] = s
                out_p[s, a, k] = base[k]
                out_bump[s, a, k] = 1.0
                pb += base[k]
            end
        end
        out_s[s, a, 4] = crash
        out_p[s, a, 4] = pc
        pcrash[s, a] = pc
        cbar[s, a] = terrain[r0+1, c0+1] + c_bump * pb
    end

    for a in 1:NACT
        out_s[goal, a, :] .= 0
        out_p[goal, a, :] .= 0.0
        out_s[goal, a, 1] = goal
        out_p[goal, a, 1] = 1.0
        cbar[goal, a] = 0.0
        out_s[crash, a, 1] = crash
        out_p[crash, a, 1] = 1.0
        cbar[crash, a] = 0.0
        avail[crash, a] = true
    end
    pcrash[goal, :] .= 0.0
    succ[crash, :] .= 0
    succ[goal, :] .= 0

    return NavSSP(n, eps, c_bump, c_crash, c_to, horizon, c_min, cost_noise,
                  grid, terrain, haz, cells, sidx, S, NS, crash, start, goal,
                  succ, avail, out_s, out_p, out_bump, cbar, pcrash)
end

# ---------------- POMDPs.jl interface ----------------

POMDPs.states(m::NavSSP) = 1:m.NS
POMDPs.actions(m::NavSSP) = 1:NACT
POMDPs.stateindex(::NavSSP, s::Int) = s
POMDPs.actionindex(::NavSSP, a::Int) = a
POMDPs.initialstate(m::NavSSP) = Deterministic(m.start)
POMDPs.isterminal(m::NavSSP, s::Int) = (s == m.goal) || (s == m.crash)
POMDPs.discount(::NavSSP) = 1.0

function POMDPs.transition(m::NavSSP, s::Int, a::Int)
    outs = Int[]
    ps = Float64[]
    @inbounds for k in 1:MAXOUT
        p = m.out_p[s, a, k]
        if p > 0.0
            push!(outs, m.out_s[s, a, k])
            push!(ps, p)
        end
    end
    return SparseCat(outs, ps)
end

"POMDPs.jl maximizes reward, so this returns the negative traversal cost."
POMDPs.reward(m::NavSSP, s::Int, a::Int) = -m.cbar[s, a]

# ---------------- exact evaluation ----------------

"""
Expected horizon truncated cost of a stationary policy. The crash state carries
the known dead end penalty `c_crash` and the goal carries zero.
"""
function eval_policy(m::NavSSP, pi::Vector{Int})
    V = fill(m.c_to, m.NS)
    V[m.goal] = 0.0
    V[m.crash] = m.c_crash
    Vn = similar(V)
    for _ in 1:m.horizon
        @inbounds for s in 1:m.NS
            a = pi[s]
            acc = m.cbar[s, a]
            for k in 1:MAXOUT
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

"Horizon truncated optimal value and its greedy stationary policy."
function optimal_value(m::NavSSP)
    V = fill(m.c_to, m.NS)
    V[m.goal] = 0.0
    V[m.crash] = m.c_crash
    Vn = similar(V)
    for _ in 1:m.horizon
        @inbounds for s in 1:m.NS
            best = Inf
            for a in 1:NACT
                acc = m.cbar[s, a]
                for k in 1:MAXOUT
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
        for a in 1:NACT
            acc = m.cbar[s, a]
            for k in 1:MAXOUT
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

"Exact success, collision and timeout probabilities from the start state."
function outcome_rates(m::NavSSP, pi::Vector{Int})
    f = zeros(m.NS)
    g = zeros(m.NS)
    f[m.goal] = 1.0
    g[m.crash] = 1.0
    fn, gn = similar(f), similar(g)
    for _ in 1:m.horizon
        @inbounds for s in 1:m.NS
            a = pi[s]
            af, ag = 0.0, 0.0
            for k in 1:MAXOUT
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

"""
Smallest value decrease along a positive probability transition of `pi`, and
the fraction of states at which every such transition decreases the value. A
nonnegative margin means the policy is consistently improving, which is the
condition under which one pass label setting is exact.
"""
function causality_margin(m::NavSSP, V::Vector{Float64}, pi::Vector{Int})
    worst = Inf
    okc, tot = 0, 0
    for s in 1:m.S
        s == m.goal && continue
        a = pi[s]
        good = true
        for k in 1:MAXOUT
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

"""
Reduced costs `w(s,a) = Q(s,a) - V(sigma(s,a))` on the determinized map.
Entries with no determinized successor are `Inf`.
"""
function reduced_costs(m::NavSSP, V::Vector{Float64})
    w = fill(Inf, m.NS, NACT)
    @inbounds for s in 1:m.NS, a in 1:NACT
        sp = m.succ[s, a]
        sp == 0 && continue
        q = m.cbar[s, a]
        for k in 1:MAXOUT
            p = m.out_p[s, a, k]
            p > 0.0 && (q += p * V[m.out_s[s, a, k]])
        end
        w[s, a] = q - V[sp]
    end
    return w
end

end # module
