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
Modules = [DORA.Learners]
```

## Warehouse benchmark domain

```@autodocs
Modules = [DORA.NavSSPs]
```

## Dijkstra oracle

```@autodocs
Modules = [DORA.Dijkstra]
```

## Reproducible RNG

```@autodocs
Modules = [DORA.RNGs]
```
