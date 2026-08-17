"""
Binary heap Dijkstra on the determinized navigation graph.

The search runs backward from the goal, so the label of a state is its
cost-to-goal and the parent pointer is the greedy action. Ties are broken by
the smaller action index, which makes the returned tree deterministic.

The counter `OPS` records edge scans. It is the implementation independent
measure of planner work reported in the paper.
"""
module Dijkstra

export dijkstra_to_goal, dijkstra_policy, ReverseAdj, reset_ops!, edge_scans

const OPS = Ref(0)
reset_ops!() = (OPS[] = 0)
edge_scans() = OPS[]

"""
Reverse adjacency of the determinized map. The geometry is static, so this is
built once and reused by every planning call.
"""
struct ReverseAdj
    start::Vector{Int}      # CSR offsets, length nstates + 1
    from::Vector{Int}       # tail state of each incoming edge
    act::Vector{Int}        # action label of each incoming edge
end

function ReverseAdj(succ::Matrix{Int})
    ns, na = size(succ)
    cnt = zeros(Int, ns + 1)
    @inbounds for s in 1:ns, a in 1:na
        sp = succ[s, a]
        sp > 0 && (cnt[sp+1] += 1)
    end
    start = similar(cnt)
    start[1] = 1
    @inbounds for i in 1:ns
        start[i+1] = start[i] + cnt[i+1]
    end
    fill = copy(start)
    from = zeros(Int, start[end] - 1)
    act = zeros(Int, start[end] - 1)
    @inbounds for s in 1:ns, a in 1:na
        sp = succ[s, a]
        if sp > 0
            from[fill[sp]] = s
            act[fill[sp]] = a
            fill[sp] += 1
        end
    end
    return ReverseAdj(start, from, act)
end

# ---------------- indexed binary min heap with decrease key ----------------

mutable struct IndexedHeap
    key::Vector{Float64}
    heap::Vector{Int}
    pos::Vector{Int}          # 0 means absent, -1 means popped
    n::Int
end

IndexedHeap(ns::Int) = IndexedHeap(fill(Inf, ns), Int[], zeros(Int, ns), 0)

@inline function isless_node(h::IndexedHeap, i::Int, j::Int)
    a, b = h.heap[i], h.heap[j]
    ka, kb = h.key[a], h.key[b]
    return ka == kb ? a < b : ka < kb
end

@inline function swapnodes!(h::IndexedHeap, i::Int, j::Int)
    h.heap[i], h.heap[j] = h.heap[j], h.heap[i]
    h.pos[h.heap[i]] = i
    h.pos[h.heap[j]] = j
end

function siftup!(h::IndexedHeap, i::Int)
    while i > 1
        p = i >> 1
        isless_node(h, i, p) || break
        swapnodes!(h, i, p)
        i = p
    end
end

function siftdown!(h::IndexedHeap, i::Int)
    while true
        l, r, m = 2i, 2i + 1, i
        l <= h.n && isless_node(h, l, m) && (m = l)
        r <= h.n && isless_node(h, r, m) && (m = r)
        m == i && return
        swapnodes!(h, i, m)
        i = m
    end
end

function pushdec!(h::IndexedHeap, s::Int, k::Float64)
    if h.pos[s] == 0
        h.key[s] = k
        push!(h.heap, s)
        h.n += 1
        h.pos[s] = h.n
        siftup!(h, h.n)
    elseif h.pos[s] > 0 && k < h.key[s]
        h.key[s] = k
        siftup!(h, h.pos[s])
    end
    return nothing
end

function popmin!(h::IndexedHeap)
    s = h.heap[1]
    last = h.heap[h.n]
    pop!(h.heap)
    h.n -= 1
    h.pos[s] = -1
    if h.n >= 1
        h.heap[1] = last
        h.pos[last] = 1
        siftdown!(h, 1)
    end
    return s, h.key[s]
end

# ---------------- the search ----------------

"""
    dijkstra_to_goal(rev, w, goal, ns, na)

Backward Dijkstra from `goal` under nonnegative weights `w[s,a]`. Returns the
cost-to-goal vector `d` and the greedy action `g[s]` that attains it. States
with no finite route keep `d = Inf` and `g = 0`.
"""
function dijkstra_to_goal(rev::ReverseAdj, w::Matrix{Float64}, goal::Int,
                          ns::Int, na::Int)
    d = fill(Inf, ns)
    g = zeros(Int, ns)
    settled = falses(ns)
    h = IndexedHeap(ns)
    d[goal] = 0.0
    pushdec!(h, goal, 0.0)

    while h.n > 0
        u, du = popmin!(h)
        settled[u] && continue
        settled[u] = true
        d[u] = du
        @inbounds for e in rev.start[u]:(rev.start[u+1]-1)
            OPS[] += 1
            s, a = rev.from[e], rev.act[e]
            settled[s] && continue
            cand = du + w[s, a]
            if cand < d[s] - 1e-15
                d[s] = cand
                g[s] = a
                pushdec!(h, s, cand)
            elseif abs(cand - d[s]) <= 1e-15 && g[s] > 0 && a < g[s]
                g[s] = a
            end
        end
    end
    return d, g
end

"""
    dijkstra_policy(rev, w, goal, avail, ns, na)

Dijkstra derived policy. States with no finite route fall back to the lowest
indexed available action so the policy is always well defined.
"""
function dijkstra_policy(rev::ReverseAdj, w::Matrix{Float64}, goal::Int,
                         avail::BitMatrix, ns::Int, na::Int)
    d, g = dijkstra_to_goal(rev, w, goal, ns, na)
    pi = ones(Int, ns)
    @inbounds for s in 1:ns
        if g[s] > 0
            pi[s] = g[s]
        else
            for a in 1:na
                if avail[s, a]
                    pi[s] = a
                    break
                end
            end
        end
    end
    pi[goal] = 1
    return pi, d
end

end # module
