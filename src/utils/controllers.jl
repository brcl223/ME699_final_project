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

mutable struct PDTracker{T}
    q::AbstractVector{<:AbstractVector{T}}
    q̇::AbstractVector{<:AbstractVector{T}}
    i::Integer
    Δt::Float64
    kp::Float64
    kd::Float64
end

function PDTracker(q, q̇; kp=100., kd=20., Δt=1e-3)
    return PDTracker(q, q̇, 1, Δt, kp, kd)
end

# LERP to find our desired values
# q̈ is just δv/δt
function calculate_desired_valuds(pd::PDTracker, ΔT::Float64, i)
    if i >= length(pd.q) - 1
        pd_i = pd.q[end]
        dim = length(pd_i)
        return pd_i, zeros(dim), zeros(dim)
    end

    qi = pd.q[i]
    q̇i = pd.q̇[i]
    q_next = pd.q[i+1]
    q̇_next = pd.q̇[i+1]

    Δt = pd.Δt

    qd_i = qi + (q_next - qi) * ΔT / Δt
    q̇d_i = q̇i + (q̇_next - q̇i) * ΔT / Δt
    q̈d_i = (q̇_next - q̇i) / Δt

    return (qd_i, q̇d_i, q̈d_i)
end

function (pd::PDTracker)(τ::AbstractVector, t, state::MechanismState)
    current_index = Integer(floor(t/pd.Δt)) + 1
    tk = (current_index - 1) * pd.Δt

    ΔT = t - tk
    q_d, q̇_d, q̈_d = calculate_desired_values(pd, ΔT, current_index)
    e = q_d - configuration(state)
    ė = q̇_d - velocity(state)

    if any(isnan,e) || any(isnan,ė)
        error("NaN during simulation.")
    end

    τ .= mass_matrix(state) * (q̈_d + pd.kp.*ė + pd.kp.*e) + dynamics_bias(state)
end
