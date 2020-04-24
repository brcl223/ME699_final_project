module Controllers

using Flux
using RigidBodyDynamics

include("./nn.jl")
using .NN

export PDController, PDGCController

function gen_rand_pi(dim)
    return 2*pi .* rand(dim)
end

mutable struct PDController
    qd::AbstractVector
    τ::AbstractVector
end

PDController(dim) = PDController(gen_rand_pi(dim), zeros(dim))

function (pd::PDController)(τ::AbstractVector, t, state::MechanismState)
    τ .= -30 .* velocity(state) - 100 * (configuration(state) - pd.qd)
    pd.τ = copy(τ)
end


# Gravity Compensated PD Controller
mutable struct PDGCController{T<:Tuple}
    qd::AbstractVector
    τ::AbstractVector
    nn::Chain{T}
end

PDGCController(dim::Integer, qd::AbstractVector) = PDGCController(qd, zeros(dim), load_nn("./models/gravity_nn"))
PDGCController(dim::Integer) = PDGCController(dim, gen_rand_pi(dim))

function (pd::PDGCController)(τ::AbstractVector, t, state::MechanismState)
    qcur = configuration(state)
    τ .= -30 .* velocity(state) - 100 * (qcur - pd.qd) + pd.nn(qcur)
end

end
