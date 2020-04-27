# TODO: Attribute controller code

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
function calculate_desired_values(pd::PDTracker, ΔT::Float64, i)
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

# Adaptive PD Controller
mutable struct ADPDController{T}
    q::AbstractVector{<:AbstractVector{T}}
    q̇::AbstractVector{<:AbstractVector{T}}
    θ̂::AbstractVector
    i::Int64
    Δt::Float64
    kp::Float64
    kd::Float64
end

function ADPDController(q, q̇; θ̂=zeros(2), kp=100., kd=100., Δt=1e-3)
    return ADPDController(q, q̇, θ̂, 1, Δt, kp, kd)
end

function calculate_desired_values(adpd::ADPDController, ΔT::Float64, i)
    if i >= length(adpd.q) - 1
        pd_i = adpd.q[end]
        dim = length(pd_i)
        return pd_i, zeros(dim), zeros(dim)
    end

    qi = adpd.q[i]
    q̇i = adpd.q̇[i]
    q_next = adpd.q[i+1]
    q̇_next = adpd.q̇[i+1]

    Δt = adpd.Δt

    qd_i = qi + (q_next - qi) * ΔT / Δt
    q̇d_i = q̇i + (q̇_next - q̇i) * ΔT / Δt
    q̈d_i = (q̇_next - q̇i) / Δt

    return (qd_i, q̇d_i, q̈d_i)
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

    ΔT = t - tk
    q_d, q̇_d, q̈_d = calculate_desired_values(adpd, ΔT, current_index)
    e = q_d - configuration(state)
    ė = q̇_d - velocity(state)
    q̈ = q̈_d + adpd.kp.*ė + adpd.kp.*e
    M̂ = mass_matrix(state);

    n = 2;
    In = Matrix(1.0I, n,n);
    On = zeros(n,n);

    Γ = 500 .*In;
    P = 0.5*[1.5*adpd.kp 0.5; 0.5 1]
    Q = [0.5*adpd.kp 0; 0 (adpd.kp+0.5)]
    A = [0 1; -adpd.kp -adpd.kp]
    B = [0 0; 1 1];

    if any(isnan,e) || any(isnan,ė)
        error("NaN during simulation.")
    end

    τ .= [200*e[1]+60*ė[1]; (M̂[2:end,2:end] * q̈[2:end] + Y(q,q̇,q̈)*adpd.θ̂)];
    Φ = M̂[2:end,2:end] \ Y(q,q̇,q̈)
    adpd.θ̂ .= -Γ \ (Φ' * (B' * P * e[2:end]))
end

# Adaptive PD Interial Based Controller
mutable struct ADPDInertial{T}
    q::AbstractVector{<:AbstractVector{T}}
    q̇::AbstractVector{<:AbstractVector{T}}
    θ̂::AbstractVector
    i::Int64
    Δt::Float64
    kp::Float64
    kd::Float64
end

function ADPDInertial(q, q̇; θ̂=zeros(2), kp=100., kd=100., Δt=1e-3)
    return ADPDInertial(q, q̇, θ̂, 1, Δt, kp, kd)
end

function calculate_desired_values(adi::ADPDInertial, ΔT::Float64, i)
    if i >= length(adi.q) - 1
        pd_i = adi.q[end]
        dim = length(pd_i)
        return pd_i, zeros(dim), zeros(dim)
    end

    qi = adi.q[i]
    q̇i = adi.q̇[i]
    q_next = adi.q[i+1]
    q̇_next = adi.q̇[i+1]

    Δt = adi.Δt

    qd_i = qi + (q_next - qi) * ΔT / Δt
    q̇d_i = q̇i + (q̇_next - q̇i) * ΔT / Δt
    q̈d_i = (q̇_next - q̇i) / Δt

    return (qd_i, q̇d_i, q̈d_i)
end

# Regressor function attributed to "Robot Manipulator Control Theory and Practice",
# Pg. 348, Example 6.3-1
function Y(q̈d, q̇d, qd, q, q̇, Λ)
    l1 = 1;
    l2 = 1;
    g = 9.81;
    c1 = cos(q[2]);
    c2 = cos(q[3]);
    c12 = cos(q[2]+q[3]);
    s1 = sin(q[2]);
    s2 = sin(q[3]);
    s12 = sin(q[2]+q[3]);

    e = qd - q;
    ė = q̇d - q̇;

    Y11 = l1^2*(q̈d[2] + Λ[1,1]*ė[2]) + l1*g*c1;
    Y12 = (l2^2 + 2*l1*l2*c2 + l1^2)*(q̈d[2] + Λ[1,1]*ė[2]) + (l2^2 + l1*l2*c2)*(q̈d[3] + Λ[2,2]*ė[3]) - l1*l2*s2*q̇[3]*(q̇d[2] + Λ[1,1]*e[2]) - l1*l2*s2*(q̇[2] + q̇[3])*(q̇d[3] + Λ[2,2]*e[3]) + l2*g*c12 +l1*g*c1;
    Y22 = (l1*l2*c2 + l2^2)*(q̈d[2] + Λ[1,1]*ė[2]) + l2^2*(q̈d[3] + Λ[2,2]*ė[3]) - l1*l2*s2*q̇[2]*(q̇d[2] + Λ[1,1]*e[2]) + l2*g*c12;
    return [Y11 Y12; 0 Y22];
end

# Control adapted from "Robot Manipulator Control Theory and Practice",
# Pg. 347, Example 6.3-1
function (adi::ADPDInertial)(τ::AbstractVector, t, state::MechanismState)

    current_index = Integer(floor(t/adi.Δt)) + 1
    tk = (current_index - 1) * adi.Δt

    q = configuration(state);
    q̇ = velocity(state);

    ΔT = t - tk
    q_d, q̇_d, q̈_d = calculate_desired_values(adi, ΔT, current_index)
    e = q_d - q
    ė = q̇_d - q̇

    n = 2;
    In = Matrix(1.0I, n,n);
    Γ = 500 .*In;
    Λ = 2.5 .*In;

    if any(isnan,e) || any(isnan,ė)
        error("NaN during simulation.")
    end

    Φ = Y(q̈_d, q̇_d, q_d, q, q̇, Λ);
    τ .= [200*e[1]+60*ė[1]; -(Φ*adi.θ̂ + adi.kd*ė[2:end] + adi.kd*Λ*e[2:end])];
    adi.θ̂ .= Γ * Φ' * (Λ*e[2:end] + ė[2:end]);
end
