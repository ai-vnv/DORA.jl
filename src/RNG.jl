"""
SplitMix64. A small deterministic generator used so that the Julia results
reproduce the Python reference implementation exactly. Every random draw in the
experiments goes through this type, in a fixed order.
"""
module RNGs

export SplitMix64, nextu64, rand01, randint, uniform, categorical

mutable struct SplitMix64
    s::UInt64
end
SplitMix64(seed::Integer) = SplitMix64(UInt64(seed))

@inline function nextu64(r::SplitMix64)
    r.s += 0x9E3779B97F4A7C15
    z = r.s
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    return z ⊻ (z >> 31)
end

"Uniform in [0,1) from the top 53 bits."
@inline rand01(r::SplitMix64) = Float64(nextu64(r) >> 11) * (1.0 / 9007199254740992.0)

"Uniform integer in 0:n-1."
@inline randint(r::SplitMix64, n::Integer) = Int(nextu64(r) % UInt64(n))

@inline uniform(r::SplitMix64, lo::Real, hi::Real) = lo + (hi - lo) * rand01(r)

"Index in 1:length(p) sampled by scanning p in order."
function categorical(r::SplitMix64, p::AbstractVector{Float64})
    u = rand01(r)
    acc = 0.0
    @inbounds for i in eachindex(p)
        acc += p[i]
        if u < acc
            return i
        end
    end
    return length(p)
end

end # module
