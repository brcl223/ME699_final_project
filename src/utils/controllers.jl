function gen_rand_pi(dim)
    return 2*pi .* rand(dim)
end

# A pseudo controller to computer q̈ given a desired
# gravity compensated torque τ
mutable struct MassController
    dim::Integer
    λ::Float64
    τ::AbstractVector
end

# This will make dim controllers to allow the same point to be tested
# With the formulation for the mass matrix, if Τ = [τ₁ .. τₙ], where each
# τᵢ has an entry of λ at i and 0 everywhere else in the vector, we get
# that Τ = λ*I, making the final computation for the mass matrix
# M(q) = λ*inv(Q̈)
function make_mass_controllers(dim::Integer, λ::Float64)
    qd = gen_rand_pi(dim)
    controllers = MassController[]
    for i = 1:dim
        τ = zeros(dim)
        τ[i] = λ
        push!(controllers, MassController(dim, λ, τ))
    end

    return controllers
end

# For this to work, ensure that the state velocities are zeroed before hand
function (mc::MassController)(τ::AbstractVector, t, state::MechanismState)
    # dynamics_bias here will be C(q,q̇)q̇ + G(q), but since we only want
    # the first state when q̇ ≈ 0, it should simply to only the gravity vector
    τ .= dynamics_bias(state) + mc.τ
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
    mass_nn::Union{Nothing,Chain}
    kfs::Union{Nothing,KalmanFilterState}
    kf_error
    qr
end

function PDTracker(q, q̇, nn=nothing, kfs=nothing; kp=100., kd=20., Δt=1e-3)
    return PDTracker(q, q̇, 1, Δt, kp, kd, nn, kfs, [], [])
end

# LERP to find our desired values
# q̈ is just δv/δt
function calculate_desired_values(q::AbstractArray,
                                  q̇::AbstractArray,
                                  Δt::Float64,
                                  ΔT::Float64,
                                  i::Int64)
    if i >= length(q) - 1
        pd_i = q[end]
        dim = length(pd_i)
        return pd_i, zeros(dim), zeros(dim)
    end

    qi = q[i]
    q̇i = q̇[i]
    q_next = q[i+1]
    q̇_next = q̇[i+1]

    qd_i = qi + (q_next - qi) * ΔT / Δt
    q̇d_i = q̇i + (q̇_next - q̇i) * ΔT / Δt
    q̈d_i = (q̇_next - q̇i) / Δt

    return (qd_i, q̇d_i, q̈d_i)
end

function combine_joint_state(q::AbstractVector, q̇::AbstractVector)
    out = []
    n = length(q)
    for i = 1:n
        push!(out,q[i])
        push!(out,q̇[i])
    end
    return out
end

function split_joint_state(x̂::AbstractVector)
    q = []
    q̇ = []
    for i = 1:length(x̂)
        if i % 2 == 0 # Even case
            push!(q̇, x̂[i])
        else
            push!(q, x̂[i])
        end
    end
    return (q, q̇)
end

# PD Tracking algorithm attributed to "Robot Manipulator Control Theory and Practice",
# Pg. 204, Table 4.4.1
function (pd::PDTracker)(τ::AbstractVector, t, state::MechanismState)
    current_index = Integer(floor(t/pd.Δt)) + 1
    tk = (current_index - 1) * pd.Δt

    q = configuration(state)
    q̇ = velocity(state)

    if pd.kfs != nothing
        N = D.Normal(0,0.1)
        qr = q + rand(N,3)
        q̇r = q̇ + rand(N,3)
        z = combine_joint_state(qr,q̇r)
        x̂ = update_filter!(pd.kfs,
                           z,
                           get_state(pd.kfs, "accel"),
                           pd.Δt,
                           current_index)

        q_e, q̇_e = split_joint_state(x̂)

        push!(pd.kf_error, vcat(q,q̇) - vcat(q_e,q̇_e))
        push!(pd.qr, qr)

        q, q̇ = (q_e, q̇_e)
    end

    ΔT = t - tk
    q_d, q̇_d, q̈_d = calculate_desired_values(pd.q, pd.q̇, pd.Δt, ΔT, current_index)
    e = q_d - q
    ė = q̇_d - q̇

    if any(isnan,e) || any(isnan,ė)
        error("NaN during simulation.")
    end

    M = if pd.mass_nn != nothing
        vec_to_sym(pd.mass_nn(q))
    else
        mass_matrix(state)
    end

    τ .= M * (q̈_d + pd.kp.*ė + pd.kp.*e) + dynamics_bias(state)

    if pd.kfs != nothing
        result = DynamicsResult(state.mechanism)
        dynamics!(result, state, τ)
        q̈_next = copy_segvec(result.v̇)
        add_state!(pd.kfs, "accel", q̈_next)
    end
end

# Adaptive PD Controller
mutable struct ADPDController{T}
    q::AbstractVector{<:AbstractVector{T}}
    q̇::AbstractVector{<:AbstractVector{T}}
    θ̂::AbstractVector
    i::Int64
    Δt::Float64
    kp::Float64
    kd::Float64
    kfs::Union{Nothing,KalmanFilterState}
    kf_error
    qr
end

function ADPDController(q, q̇, kfs=nothing; θ̂=zeros(2), kp=100., kd=100., Δt=1e-3)
    return ADPDController(q, q̇, θ̂, 1, Δt, kp, kd, kfs, [], [])
end

# Regressor function attributed to "Robot Manipulator Control Theory and Practice",
# Pg. 332, Example 6.2-1
function Y(q, q̇, q̈)
    l1 = 1;
    l2 = 1;
    g = 9.81;
    c1 = cos(q[2]);
    c2 = cos(q[3]);
    c12 = cos(q[2]+q[3]);
    s1 = sin(q[2]);
    s2 = sin(q[3]);
    s12 = sin(q[2]+q[3]);

    Y11 = l1^2*q̈[2] + l1*g*c1;
    Y12 = l2^2*(q̈[2]+q̈[2]) + l1*l2*c2*(2*q̈[2]+q̈[3])+l1^2*q̈[2]-l1*l2*s2*q̇[3]^2 - 2*l1*l2*s2*q̇[2]*q̇[3] + l2*g*c12 + l1*g*c1;
    Y22 = l1*l2*c2*q̈[2] + l1*l2*s2*q̇[2]^2 + l2*g*c12 + l2^2*(q̈[2]+q̈[3]);
    return [Y11 Y12; 0 Y22];
end

# Control adapted from "Robot Manipulator Control Theory and Practice",
# Pgs. 339-341, Example 6.2-2
function (adpd::ADPDController)(τ::AbstractVector, t, state::MechanismState)

    current_index = Integer(floor(t/adpd.Δt)) + 1
    tk = (current_index - 1) * adpd.Δt

    q = configuration(state);
    q̇ = velocity(state);

    if adpd.kfs != nothing
        N = D.Normal(0,0.1)
        qr = q + rand(N,3)
        q̇r = q̇ + rand(N,3)
        z = combine_joint_state(qr,q̇r)
        x̂ = update_filter!(adpd.kfs,
                           z,
                           get_state(adpd.kfs, "accel"),
                           adpd.Δt,
                           current_index)

        q_e, q̇_e = split_joint_state(x̂)

        push!(adpd.kf_error, vcat(q,q̇) - vcat(q_e,q̇_e))
        push!(adpd.qr, copy(qr))

        q, q̇ = (q_e, q̇_e)
    end

    ΔT = t - tk
    q_d, q̇_d, q̈_d = calculate_desired_values(adpd.q, adpd.q̇, adpd.Δt, ΔT, current_index)
    e = q_d - configuration(state)
    ė = q̇_d - velocity(state)
    q̈ = q̈_d + adpd.kp.*ė + adpd.kp.*e
    M̂ = mass_matrix(state)

    n = 2;
    In = Matrix(1.0I, n,n);
    On = zeros(n,n);

    Γ = 500 .*In;
    P = 0.5*[1.5*adpd.kp 0.5; 0.5 1]
    Q = [0.5*adpd.kp 0; 0 (adpd.kp+0.5)]
    A = [0 1; -adpd.kp -adpd.kp]
    B = [0 0; 1 1];

    if any(isnan, q)
        error("NaN during simulation: q")
    end

    if any(isnan, q̇)
        error("NaN during simulation: q̇")
    end

    if any(isnan, q_d)
        error("NaN during simulation: q_d")
    end

    if any(isnan, q̇_d)
        error("NaN during simulation: q̇_d")
    end

    if any(isnan, q̈_d)
        error("NaN during simulation: q̈_d")
    end

    if any(isnan, M̂)
        error("NaN during simulation: M̂")
    end

    if any(isnan,e) || any(isnan,ė)
        error("NaN during simulation: Error estimate")
    end

    τ .= [200*e[1]+60*ė[1]; (M̂[2:end,2:end] * q̈[2:end] + Y(q,q̇,q̈)*adpd.θ̂)];
    Φ = M̂[2:end,2:end] \ Y(q,q̇,q̈)
    adpd.θ̂ .= -Γ \ (Φ' * (B' * P * e[2:end]))
end
