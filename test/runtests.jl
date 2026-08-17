using Test
using DORASolvers
using POMDPs
using POMDPModels
using POMDPTools
using Random

# A minimal MDP that only defines the four-argument reward arity, to exercise
# the solver's expectation fallback. State 2 is a dead end whose every action
# leads to the bad terminal 4; state 3 reaches the good terminal 5.
struct ChainMDP <: MDP{Int,Int} end
POMDPs.actions(::ChainMDP) = [1, 2]
POMDPs.discount(::ChainMDP) = 1.0
POMDPs.isterminal(::ChainMDP, s::Int) = s == 4 || s == 5
POMDPs.initialstate(::ChainMDP) = Deterministic(1)
function POMDPs.transition(::ChainMDP, s::Int, a::Int)
    s == 1 && return Deterministic(a == 1 ? 3 : 2)
    s == 2 && return Deterministic(4)
    s == 3 && return Deterministic(5)
    return Deterministic(s)              # terminals self-loop
end
POMDPs.reward(::ChainMDP, s::Int, a::Int, sp::Int) =
    sp == 5 ? 10.0 : sp == 4 ? -10.0 : -1.0

@testset "DORASolvers.jl" begin

@testset "SplitMix64 reproducibility" begin
    # Hardcoded values cross-validated against ref/rng.py (seed 1234).
    r = SplitMix64(1234)
    @test [rand01(r) for _ in 1:4] ≈ [0.730666524540624, 0.5928898580149862,
                                      0.20213287431010984, 0.3061867920503709]
    r = SplitMix64(1234)
    @test DORASolvers.RNGs.nextu64(r) == 0xbb0cf61b2f181cdb
    @test DORASolvers.RNGs.nextu64(r) == 0x97c7a1364df06524
    r = SplitMix64(7)
    @test all(0 .<= [randint(r, 5) for _ in 1:20] .<= 4)
    @test categorical(SplitMix64(3), [0.0, 0.0, 1.0]) == 3
end

@testset "Dijkstra on a hand-built graph" begin
    # Line 1 -> 2 -> 3 -> 4(goal) with action 1, plus a direct edge 1 -> 4
    # with action 2. Weights make the line cost 3 and the shortcut cost 10.
    succ = [2 4; 3 0; 4 0; 0 0]
    avail = trues(4, 2)
    w = [1.0 10.0; 1.0 1e6; 1.0 1e6; 1e6 1e6]
    rev = ReverseAdj(succ)
    reset_ops!()
    d, g = dijkstra_to_goal(rev, w, 4, 4, 2)
    @test d ≈ [3.0, 2.0, 1.0, 0.0]
    @test g[1:3] == [1, 1, 1]
    @test edge_scans() > 0

    # Make the shortcut match the line exactly: tie broken by smaller action.
    w2 = [1.0 3.0; 1.0 1e6; 1.0 1e6; 1e6 1e6]
    _, g2 = dijkstra_to_goal(rev, w2, 4, 4, 2)
    @test g2[1] == 1

    pi, dp = dijkstra_policy(rev, w, 4, avail, 4, 2)
    @test pi[1:3] == [1, 1, 1]
    @test dp ≈ d
end

rewards = Dict(GWPos(4, 3) => -10.0, GWPos(4, 6) => -10.0,
               GWPos(9, 3) => 10.0)
mdp = SimpleGridWorld(size=(10, 10), rewards=rewards, tprob=0.7)

@testset "tabularize on SimpleGridWorld" begin
    classify = sp -> begin
        r = get(rewards, sp, 0.0)
        r > 0.0 ? :goal : r < 0.0 ? :crash : :normal
    end
    m = tabularize(mdp; start=GWPos(1, 1), classify=classify,
                   cost=(s, a, sp) -> 1.0, c_min=0.5, c_to=30.0,
                   c_crash=20.0, horizon=100, cost_noise=0.2,
                   name="gridworld")
    @test m.NS == m.S + 2
    @test m.S == 97          # 100 cells minus one goal and two hazards
    @test m.start == 1
    @test m.states[1] == GWPos(1, 1)
    # probability mass of every ordinary state action pair is one
    for s in 1:m.S, a in 1:m.NA
        @test sum(m.out_p[s, a, :]) ≈ 1.0
    end
    # goal and crash are absorbing
    @test m.out_s[m.goal, 1, 1] == m.goal
    @test m.out_s[m.crash, 1, 1] == m.crash
    # optimal value of the start is finite and below the timeout penalty
    V, pistar = optimal_value(m)
    @test 0.0 < V[m.start] < m.c_to
end

@testset "DORASolver known-costs default" begin
    sol = DORASolver(start=GWPos(1, 1))
    p = @test_logs (:warn,) match_mode = :any solve(sol, mdp)
    @test p isa DORAPlanner
    a = action(p, GWPos(1, 1))
    @test a in actions(mdp)
    @test oracle_calls(p.learner) == sol.iters

    # the planner policy is near optimal on the tabularized model
    V, _ = optimal_value(p.tab)
    Jstar = V[p.tab.start]
    Jp = eval_policy(p.tab, p.pi)[p.tab.start]
    @test (Jp - Jstar) / Jstar < 0.01

    # value follows the POMDPs.jl maximization convention
    @test value(p, GWPos(1, 1)) < 0.0
    # states collapsed into a sink fall back gracefully
    @test action(p, GWPos(9, 3)) in actions(mdp)
    @test value(p, GWPos(9, 3)) == 0.0

    # the planner runs under the standard POMDPTools simulator
    r = simulate(RolloutSimulator(rng=MersenneTwister(7), max_steps=100), mdp, p)
    @test r isa Float64
end

# Regression: the evaluation helpers are exported by DORASolvers, and used to
# be shadowed by same-named but disconnected functions inside TabularSSPs, so
# `using DORASolvers; optimal_value(planner.tab)` raised a MethodError. They
# must be one generic per name, carrying methods for both model types.
# Placed after the testset above, which asserts on the once-per-session
# `maxlog = 1` discount warning and therefore has to see the first `solve`.
@testset "exported evaluation API accepts a TabularSSP" begin
    classify = sp -> begin
        r = get(rewards, sp, 0.0)
        r > 0.0 ? :goal : r < 0.0 ? :crash : :normal
    end
    tab = tabularize(mdp; start=GWPos(1, 1), classify=classify,
                     cost=(s, a, sp) -> 1.0, c_min=0.5, c_to=30.0,
                     c_crash=20.0, horizon=100, cost_noise=0.0)

    # one function object per name, shared by the warehouse and tabular models
    for f in (eval_policy, optimal_value, outcome_rates, causality_margin,
              reduced_costs)
        nm = nameof(f)
        @test f === getfield(DORASolvers.NavSSPs, nm)
        @test f === getfield(DORASolvers.TabularSSPs, nm)
    end
    @test hasmethod(optimal_value, Tuple{TabularSSP})
    @test hasmethod(eval_policy, Tuple{TabularSSP,Vector{Int}})
    @test hasmethod(outcome_rates, Tuple{TabularSSP,Vector{Int}})
    @test hasmethod(causality_margin,
                    Tuple{TabularSSP,Vector{Float64},Vector{Int}})
    @test hasmethod(reduced_costs, Tuple{TabularSSP,Vector{Float64}})
    # the TabularSSP methods really are the ones defined in TabularSSPs
    @test which(optimal_value, Tuple{TabularSSP}).module === DORASolvers.TabularSSPs

    # backward compatibility: the module-qualified path that external scripts
    # used before the fix still resolves and still returns the same result,
    # even though the methods now live on the NavSSPs generics
    @test DORASolvers.TabularSSPs.optimal_value(tab) == optimal_value(tab)
    @test DORASolvers.NavSSPs.optimal_value(tab) == optimal_value(tab)

    # and they run, called through the package-level names only
    V, pistar = optimal_value(tab)
    @test 0.0 < V[tab.start] < tab.c_to
    @test eval_policy(tab, pistar)[tab.start] ≈ V[tab.start] atol = 1e-6
    sr, cr, tr = outcome_rates(tab, pistar)
    @test sr + cr + tr ≈ 1.0 atol = 1e-9
    marg, causf = causality_margin(tab, V, pistar)
    @test isfinite(marg) && 0.0 <= causf <= 1.0
    @test count(isfinite, reduced_costs(tab, V)) > 0

    # the documented path: analyse the model behind a planner
    planner = solve(DORASolver(start=GWPos(1, 1)), mdp)
    action(planner, GWPos(1, 1))
    Vp, _ = optimal_value(planner.tab)
    Jstar = Vp[planner.tab.start]
    Jp = eval_policy(planner.tab, planner.pi)[planner.tab.start]
    # the DORA policy can only be at least as costly as the exact optimum,
    # and the documented bound on that excess is 1% (see docs/src/examples.md)
    @test Jp >= Jstar - 1e-9
    @test (Jp - Jstar) / Jstar < 0.01
end

@testset "DORASolver online training" begin
    sol = DORASolver(start=GWPos(1, 1), known_costs=false, train_episodes=300,
                     seed=1)
    p = solve(sol, mdp)
    @test oracle_calls(p.learner) == sol.train_episodes * sol.iters
    V, _ = optimal_value(p.tab)
    Jstar = V[p.tab.start]
    Jp = eval_policy(p.tab, replan!(p))[p.tab.start]
    @test (Jp - Jstar) / Jstar < 0.05

    # train! continues learning on the same planner
    n0 = oracle_calls(p.learner)
    train!(p, 10)
    @test oracle_calls(p.learner) == n0 + 10 * sol.iters
    @test p.dirty
end

@testset "DORAPlanner online interface" begin
    p = solve(DORASolver(start=GWPos(1, 1), known_costs=false), mdp)
    a = action(p, GWPos(1, 1))       # triggers the first lazy plan
    @test !p.dirty
    n0 = oracle_calls(p.learner)
    observe!(p, GWPos(1, 1), a, 1.3)
    @test p.dirty
    action(p, GWPos(1, 1))           # lazy replan
    @test !p.dirty
    @test oracle_calls(p.learner) == n0 + p.learner.iters
    @test p.learner.st.N[1, findfirst(isequal(a), p.acts)] == 1.0

    @test_throws ArgumentError observe!(p, GWPos(9, 3), a, 1.0)
    @test_throws ArgumentError observe!(p, GWPos(1, 1), :not_an_action, 1.0)
end

@testset "RNG edge cases" begin
    # first draw of seed 1234 is 0.7307 > 0.6: the scan runs off the end
    @test categorical(SplitMix64(1234), [0.3, 0.3]) == 2
    @test uniform(SplitMix64(1), 2.0, 2.0) == 2.0
end

@testset "NavSSP warehouse" begin
    m = build(; n=20, eps=0.10, p_haz=0.14, rough=0.5, c_crash=80.0,
              c_to=60.0, horizon=150, cost_noise=0.12,
              terrain_rng=SplitMix64(7000))
    @test m.NS == m.S + 1
    @test m.crash == m.NS
    @test m.goal != m.start

    # POMDPs.jl interface
    @test states(m) == 1:m.NS
    @test actions(m) == 1:NACT
    @test stateindex(m, 3) == 3
    @test actionindex(m, 2) == 2
    @test rand(MersenneTwister(1), initialstate(m)) == m.start
    @test isterminal(m, m.goal) && isterminal(m, m.crash)
    @test !isterminal(m, m.start)
    @test Base.invokelatest(discount, m) == 1.0   # not folded away
    d = transition(m, m.start, 1)
    @test sum(p for (sp, p) in weighted_iterator(d)) ≈ 1.0
    @test reward(m, m.start, 1) == -m.cbar[m.start, 1]

    # exact evaluation helpers
    V, pistar = optimal_value(m)
    @test 0.0 < V[m.start] < m.c_to
    @test eval_policy(m, pistar)[m.start] ≈ V[m.start] atol = 1e-6
    sr, cr, tr = outcome_rates(m, pistar)
    @test sr + cr + tr ≈ 1.0 atol = 1e-9
    @test sr > 0.9
    marg, causf = causality_margin(m, V, pistar)
    @test isfinite(marg)
    @test 0.0 <= causf <= 1.0
    ws = reduced_costs(m, V)
    fin = isfinite.(ws)
    @test count(fin) > 0
    @test minimum(ws[fin]) > -1.0

    # build variants: no roughness rng, no hazard band
    m0 = build(; n=16, p_haz=0.0, rough=0.0)
    @test all(m0.haz .== 0.0)
end

# The shelf pattern fills rows whose zero-based index is 2 or 3 modulo 4, and
# build places the start and the goal on row n - 12, so only n >= 12 with
# n % 4 in (0, 1) yields free start and goal cells. Other sizes used to reach
# the array setup with a zero index and fail with a BoundsError (n = 18), or
# return a model whose start index was silently 0 (n = 14).
@testset "build grid size validation" begin
    for n in (12, 13, 16, 17, 20, 21, 24, 25)
        m = build(; n=n, p_haz=0.0, rough=0.0)
        # both resolved to ordinary cells, never to the 0 index that used to
        # slip through, and never to the crash index m.NS
        @test 1 <= m.start <= m.S
        @test 1 <= m.goal <= m.S
        @test m.start != m.goal
    end

    # below the minimum: row n - 12 is above the top border
    for n in (0, 1, 8, 11)
        @test_throws ArgumentError build(; n=n)
    end

    # start or goal would land inside a shelf row
    for n in (14, 15, 18, 19, 22, 23)
        @test_throws ArgumentError build(; n=n)
    end

    # boundaries of the accepted residue classes, either side of a valid size
    @test_throws ArgumentError build(; n=11)   # just below the minimum
    @test build(; n=12) isa NavSSP             # the minimum itself
    @test build(; n=13) isa NavSSP             # last accepted residue
    @test_throws ArgumentError build(; n=14)   # first rejected residue

    # the messages name the offending argument, not an internal index
    err = try
        build(; n=18)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("n = 18", err.msg)
end

# Roughness needs a stream; asking for it without one used to be ignored.
@testset "build roughness requires a terrain rng" begin
    @test_throws ArgumentError build(; n=20, rough=0.5)
    @test_throws ArgumentError build(; n=20, rough=0.5, terrain_rng=nothing)

    # unchanged behaviour for the two supported combinations
    flat = build(; n=20, rough=0.0)
    rugged = build(; n=20, rough=0.5, terrain_rng=SplitMix64(7000))
    @test rugged.terrain != flat.terrain           # the field really is added
    @test build(; n=20, rough=0.0, terrain_rng=nothing) isa NavSSP
end

@testset "learner zoo on the warehouse" begin
    m = build(; n=20, eps=0.10, p_haz=0.14, rough=0.5,
              terrain_rng=SplitMix64(7000))
    K = 25
    ctors = [
        ("DORA", () -> DORALearner(m, K)),
        ("DORA-0", () -> DORA0(m, K)),
        ("CED", () -> CED(m, K)),
        ("EGD", () -> EGD(m, K)),
        ("OVI-K", () -> OptimisticVI(m, K; known_P=true)),
        ("OVI-U", () -> OptimisticVI(m, K; known_P=false)),
        ("SARSA", () -> Sarsa(m, K)),
        ("MCTS", () -> MCTSPlan(m, K; sims=15, depth=15)),
    ]
    for (name, ctor) in ctors
        L = ctor()
        rng = SplitMix64(31000)
        for _ in 1:(L isa MCTSPlan ? 2 : K)   # 2nd MCTS plan hits the cache
            pi = plan!(L)
            @test length(pi) == m.NS
            run_episode!(L, pi, rng)
        end
        @test work(L) >= 0
        @test oracle_calls(L) >= 0
        @test vi_sweeps(L) >= 0
    end

    # chance constrained DORA with multiplier updates
    R = RiskDORA(m, K; budget=0.05)
    rng = SplitMix64(31007)
    for _ in 1:K
        pi = plan!(R)
        _, _, crashed, _ = run_episode!(R, pi, rng)
        update_multiplier!(R, crashed)
    end
    @test R.lam >= 0.0
    @test all(0.0 .<= rhat(R) .<= 0.95)
    @test work(R) > 0 && oracle_calls(R) > 0 && vi_sweeps(R) == 0

    # a tiny horizon forces the timeout return of the warehouse episode
    mt = build(; n=20, horizon=3)
    Lt = DORALearner(mt, 2)
    _, ok, crashed, timeout = run_episode!(Lt, plan!(Lt), SplitMix64(5))
    @test timeout && !ok && !crashed

    # SARSA temporal difference branches, driven directly
    Ls = Sarsa(m, K)
    DORASolvers.Learners.sarsa_step!(Ls, m.start, 1, 1.0, m.goal)
    DORASolvers.Learners.sarsa_step!(Ls, m.start, 1, 1.0, m.crash)
    DORASolvers.Learners.sarsa_step!(Ls, m.start, 1, 1.0, m.start)
    @test Ls.Q[m.start, 1] > 0.0
end

@testset "TabularSSP evaluation helpers" begin
    classify = sp -> begin
        r = get(rewards, sp, 0.0)
        r > 0.0 ? :goal : r < 0.0 ? :crash : :normal
    end
    m = tabularize(mdp; start=GWPos(1, 1), classify=classify,
                   cost=(s, a, sp) -> 1.0, c_min=0.5, c_to=30.0,
                   c_crash=20.0, horizon=100, cost_noise=0.2)
    V, pistar = optimal_value(m)
    @test eval_policy(m, pistar)[m.start] ≈ V[m.start] atol = 1e-6
    sr, cr, tr = outcome_rates(m, pistar)
    @test sr + cr + tr ≈ 1.0 atol = 1e-9
    marg, causf = causality_margin(m, V, pistar)
    @test isfinite(marg) && 0.0 <= causf <= 1.0
    # the clamped reduced-cost member of the Dijkstra class is near optimal
    ws = reduced_costs(m, V)
    fin = isfinite.(ws)
    @test count(fin) > 0
    wr = [fin[s, a] ? max(ws[s, a], 0.0) : 1e6 for s in 1:m.NS, a in 1:m.NA]
    piR, _ = dijkstra_policy(ReverseAdj(m.succ), wr, m.goal, m.avail, m.NS, m.NA)
    JR = eval_policy(m, piR)[m.start]
    @test (JR - V[m.start]) / V[m.start] < 0.01

    # a two-step horizon forces the timeout return of the tabular episode
    mt = tabularize(mdp; start=GWPos(1, 1), classify=classify,
                    cost=(s, a, sp) -> 1.0, c_min=0.5, c_to=30.0,
                    c_crash=20.0, horizon=2, cost_noise=0.0)
    Lt = DORALearner(mt, 2)
    _, ok, crashed, timeout = run_episode!(Lt, plan!(Lt), SplitMix64(5))
    @test timeout && !ok && !crashed

    # SARSA on a state with no available action (dead end collapsed to crash)
    tabc = tabularize(ChainMDP(); start=1,
                      classify=sp -> sp == 5 ? :goal : sp == 4 ? :crash : :normal,
                      cost=(s, a, sp) -> 1.0, c_min=0.5, c_to=10.0,
                      c_crash=5.0, horizon=10, cost_noise=0.0)
    deadend = findfirst(==(2), tabc.states)
    @test all(.!tabc.avail[deadend, :])
    Lc = Sarsa(tabc, 2)
    DORASolvers.Learners.sarsa_step!(Lc, tabc.start, 1, 1.0, deadend)
    @test Lc.Q[tabc.start, 1] > 0.0
end

@testset "learner zoo on the tabular gridworld" begin
    classify = sp -> begin
        r = get(rewards, sp, 0.0)
        r > 0.0 ? :goal : r < 0.0 ? :crash : :normal
    end
    m = tabularize(mdp; start=GWPos(1, 1), classify=classify,
                   cost=(s, a, sp) -> 1.0, c_min=0.5, c_to=30.0,
                   c_crash=20.0, horizon=100, cost_noise=0.2)
    for ctor in [() -> EGD(m, 10), () -> OptimisticVI(m, 10; known_P=false),
                 () -> RiskDORA(m, 10), () -> Sarsa(m, 10)]
        L = ctor()
        rng = SplitMix64(1)
        for _ in 1:10
            run_episode!(L, plan!(L), rng)
        end
    end

    # constant-returning accessors, invoked so that constant propagation
    # cannot fold the calls away
    R2 = RiskDORA(m, 2)
    M2 = MCTSPlan(m, 2; sims=2, depth=2)
    @test Base.invokelatest(DORASolvers.Learners.vi_sweeps, R2) == 0
    @test Base.invokelatest(DORASolvers.Learners.explore, R2) == 0.0
    @test Base.invokelatest(DORASolvers.Learners.explore, M2) == 0.0
end

@testset "solver on a reward-arity-4 MDP" begin
    # terminal states classified directly (unreachable through the BFS, since
    # the entering reward cells collapse first)
    cls = DORASolvers._default_classify(ChainMDP(), [1, 2])
    @test cls(4) == :crash
    @test cls(5) == :goal

    # ChainMDP defines only reward(m, s, a, sp): the solver falls back to the
    # expected reward; defaults classify 5 as goal and both 2 and 4 as crash
    p = solve(DORASolver(), ChainMDP())
    @test action(p, 1) == 1               # avoid the dead end branch
    @test !(2 in p.tab.states)
    @test value(p, 1) < 0.0

    # explicit penalty and classification overrides
    p2 = solve(DORASolver(start=1, c_to=44.0, c_crash=11.0,
                          classify=sp -> sp == 5 ? :goal :
                                         sp == 4 ? :crash : :normal,
                          cost=(s, a, sp) -> 1.0), ChainMDP())
    @test p2.tab.c_to == 44.0
    @test p2.tab.c_crash == 11.0
    @test 2 in p2.tab.states              # kept as a normal state now
    @test action(p2, 1) == 1
end

@testset "default classify and cost derivation" begin
    # no start given: sampled from initialstate; defaults classify the
    # positive reward cell as goal and the hazards as crashes
    p = solve(DORASolver(seed=3), mdp)
    @test p.tab.S <= 97
    @test action(p, p.tab.states[1]) in actions(mdp)
    # hazard and goal cells were collapsed, not enumerated
    @test !(GWPos(9, 3) in p.tab.states)
    @test !(GWPos(4, 3) in p.tab.states)
end

end
