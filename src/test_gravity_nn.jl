using Flux
using BSON: @load
using RigidBodyDynamics

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

function main()
    @load "./models/gravity_nn.bson" nn

    urdf = joinpath(".", "robot.urdf")
    mechanism = parse_urdf(urdf)
    state = MechanismState(mechanism)
    dim = num_positions(mechanism)

    while true
        pd = PDController(dim)
        final_time = 10.
        ts, qs, vs = simulate(state, final_time, pd; Δt=1e-3)

        println("Configuration")
        @show qs[end]

        println("Desired Configuration")
        @show pd.qd

        println("Gravity Torques")
        @show pd.τ

        println("Predicted Torques")
        pred = nn(qs[end])
        @show pred

        println("MSE Error")
        @show Flux.mse(pd.τ, pred)

        readline(stdin)
    end
end



main()
