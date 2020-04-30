function bound_2pi(v::AbstractVector)
    bound = floor.(v ./ (2*pi))
    return v .- 2*pi .* bound
end

function bound_pi_npi(v::AbstractVector)
    return bound_2pi(v .+ pi) .- pi
end

function bound_2pi_n2pi(v::AbstractVector)
    return 2 * (bound_2pi(0.5 .* v .+ pi) .- pi)
end

function bound_joints(v::AbstractVector)
    # All joints bounded by -2Π < v < 2Π
    return bound_2pi_n2pi(v)
end

function gen_rand_pi(dim)
    # Generate number -2Π < x < 2Π
    return 4*pi .* (rand(dim) .- 0.5)
end

# Simple helper function to copy RigidBodyDynamics SegmentedVector
# to a more useable object
function copy_segvec(q::AbstractVector{T}) where T
    out = Float64[]
    for v in q
        push!(out, v)
    end
    return out
end

function sym_to_vec(m::AbstractArray)
    s = size(m)
    if length(s) != 2 && s[1] != s[2]
        error("ERROR: sym_to_vec: Matrix must be two dimensional square matrix")
    end

    out = []

    dim = s[1]
    for i = 1:dim
        for j = 1:i
            push!(out, m[i,j])
        end
    end

    return out
end

function vec_to_sym(v::AbstractArray; dims_to_check=5)
    dim = nothing
    for i = 1:dims_to_check
        if length(v) == sum(1:i)
            dim = i
            break
        end
    end

    if dim == nothing
        error("ERROR: vec_to_sym: Vector length does not correspond to lower square symmetric matrix")
    end

    out = zeros(dim,dim)
    for i = 1:dim
        s = sum(1:(i-1))
        for j = 1:i
            out[i,j] = v[s + j]
        end
    end
    return Symmetric(out, :L)
end
