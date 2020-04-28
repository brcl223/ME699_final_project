# Attribution for equations:
# https://www.kalmanfilter.net/default.aspx

mutable struct KalmanFilter
    x̂::AbstractVector
    F # Function of time
    G # Function of time
    H::AbstractMatrix
    P̂::AbstractMatrix
    Kₙ::AbstractMatrix
    Σ::AbstractMatrix
end

function KalmanFilter(F,
                      G,
                      H::AbstractMatrix,
                      σ::AbstractVector,
                      x₀::AbstractVector,
                      P₀::AbstractMatrix)
    Σ = Diagonal(σ * σ')
    P₀ = copy(P₀)
    x₀ = copy(x₀)
    return KalmanFilter(x₀, F, G, H, P₀, ones(reverse(size(H))), Σ)
end

function update_filter!(k::KalmanFilter, z::AbstractVector, u::AbstractVector, Δt::Float64=1e-3)
    # Update gains
    k.Kₙ = k.P̂*k.H'* inv(k.H*k.P̂*k.H' + k.Σ)

    # State update
    xₙ = k.x̂ + k.Kₙ*(z - k.H*k.x̂)

    # Next state estimate given measurement z and input reading u
    k.x̂ = k.F(Δt)*xₙ + k.G(Δt)*u

    # Calculate Covariance update
    Pₙ = (I - k.Kₙ*k.H) * k.P̂ * (I - k.Kₙ*k.H)' + k.Kₙ*k.Σ*k.Kₙ'

    # Update covariance estimate
    k.P̂ = k.F(Δt)*Pₙ*k.F(Δt)'

    return (xₙ, Pₙ)
end


mutable struct KalmanFilterState
    kf::KalmanFilter
    iter::Int64
    xₙ::AbstractVector
    state::Dict{String,Any}
end

# Kalman filter specifically designed to hold state to help
# the issue with the ODE solver
function KalmanFilterState(F,
                           G,
                           H::AbstractMatrix,
                           σ::AbstractVector,
                           x₀::AbstractVector,
                           P₀::AbstractMatrix)
    kf = KalmanFilter(F,G,H,σ,x₀,P₀)
    return KalmanFilterState(kf, 0, similar(x₀), Dict())
end

function update_filter!(k::KalmanFilterState,
                        z::AbstractVector,
                        u::AbstractVector,
                        Δt::Float64,
                        iter::Int64)
    if iter <= k.iter
        return k.xₙ
    end

    k.iter = iter
    xₙ, _ = update_filter!(k.kf, z, u, Δt)
    k.xₙ = xₙ
    return xₙ
end

function add_state!(k::KalmanFilterState,
                    key::String,
                    val)
    k.state[key] = copy(val)
end

function get_state(k::KalmanFilterState,
                   key::String)
    return k.state[key]
end

# Estimate the weight of a gold bar on a scale
# The scale produces noise in it's readings, sampled from
# e ∼ N(0,σ). Each reading consists of x̃ = x + e
#
# The state equation for this example is trivial.
# A = [0], as ẋ = 0 since the weight of the bar
# does not change
# B = [0] as no force is applied to the bar
# C = [1], as we want to measure z = x
#
# Our sensor variance will be 'var'
# So R = [var]
#
# Our initial guess x₀ will be 0
# Our initial uncertainty P₀ will be 10
function goldbar_weight_example()
    N = Normal(0,3)
    x = 10 # The true weight of the bar

    A = Matrix(ones(1,1))
    B = Matrix(zeros(1,1))
    C = Matrix(ones(1,1))
    σ = [var(N)]

    x₀ = [mean(x .+ rand(N,10))]
    P₀ = Matrix(var(N)*ones(1,1))

    kf = KalmanFilter(A,B,C,σ,x₀,P₀)

    for i = 0:100000
        new_reading = [x + rand(N)]
        input = zeros(1)
        xₙ,Pₙ = update_filter!(kf, new_reading, input)

        if i % 1 == 0
            println("Iteration $(i + 1)")
            # @show new_reading
            # @show input
            # @show xₙ
            # @show Pₙ
            readline(stdin)
        end
    end
end

# A 2D example of the Kalman filter, tracking an car
# on the road in two dimensions. It will be driving
# in a straight line with velocity (1,1), and starting
# at position (10,10)
#
# A = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0] since ẋ₁ = x₂,
#                                                ẋ₂ = 0,
#                                                ẋ₃ = x₄,
#                                                ẋ₄ = 0
# B = [0; 0; 0; 0] since there is no input
# C = I₄ since we would like to measure both
#        velocity and position
#
# Thus these values give us that
# F = I + AΔt = [1 Δt 0 0; 0 1 0 0; 0 0 1 Δt; 0 0 0 1]
# G = zeros(4,2) since B = zeros(4,1)
# H = C since we can observe both position and velocity
#
# σ = 3
# Σ = diag(σ) since the sensor values will have the same error
# x₀ = [10; 1; 10; 1]
# P₀ = diag(σ)
function constant_velocity_car_example()
    Δt = 0.001 # Sampling rate
    A = [0 1 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0]
    F = [1 Δt 0 0; 0 1 0 0; 0 0 1 Δt; 0 0 0 1]
    G = zeros(4,2)
    H = [1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.]

    N = Normal(0,0.1)
    σ = ones(4) * var(N)

    x₀ = [10.; 1.; 10.; 1.]
    P₀ = Diagonal(var(N)*ones(4,4))

    kf = KalmanFilter(F,G,H,σ,x₀,P₀)

    # Solve the actual ODE
    # Should be simple enough since δp/δt = c

    samples = zeros(4,0)
    tfinal = 1.
    n = 10000
    t = 0
    for i = 1:n
        x = x₀ + A*x₀*t
        samples = hcat(samples, x)
        t += Δt * tfinal
    end

    random = copy(samples)
    for i = 1:size(samples)[2]
        random[:,i] += rand(N,4)
        xₙ,Pₙ = update_filter!(kf, random[:,i], zeros(2))

        @show samples[:,i]
        @show random[:,i]
        @show xₙ
        @show Pₙ
        readline(stdin)
    end
end
