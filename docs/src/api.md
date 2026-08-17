# API Reference

## Solver interface

```@docs
DORASolver
DORAPlanner
replan!
train!
observe!
```

The planner also implements `POMDPs.action` and `POMDPs.value`.

## Model conversion

`solve` calls [`tabularize`](@ref) to turn the model into the `TabularSSP` it
plans on, and leaves the result in `planner.tab`. The exact evaluation helpers
documented here accept that object directly through the package-level exports,
so `optimal_value(planner.tab)` works after `using DORASolvers` alone.

```@autodocs
Modules = [DORASolvers.TabularSSPs]
```

## Learners and the episode-level API

```@autodocs
Modules = [DORASolvers.Learners]
```

## Warehouse benchmark domain

```@autodocs
Modules = [DORASolvers.NavSSPs]
```

## Dijkstra oracle

```@autodocs
Modules = [DORASolvers.Dijkstra]
```

## Reproducible RNG

```@autodocs
Modules = [DORASolvers.RNGs]
```
