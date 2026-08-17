# API Reference

## Solver interface

```@docs
DORASolver
DORAPlanner
replan!
train!
```

The planner also implements `POMDPs.action` and `POMDPs.value`, and
`observe!(planner, s, a, cost)` reports an observed traversal cost (see
below).

## Model conversion

```@docs
tabularize
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
