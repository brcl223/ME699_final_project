using RigidBodyDynamics
using JLD

const VEL_EPS = 1e-6
const NUM_SAMPLES = 5000

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
    urdf = joinpath(".", "robot.urdf")
    mechanism = parse_urdf(urdf)
    state = MechanismState(mechanism)
    dim = num_positions(mechanism)

    qpos = zeros(dim,0)
    taus = zeros(dim,0)

    for i = 1:NUM_SAMPLES
        if i % 100 == 0
            println("Currently on iteration $i")
        end

        pd = PDController(dim)
        final_time = 10.
        ts, qs, vs = simulate(state, final_time, pd; Δt=1e-3)

        bad_sim = false
        for v in vs[end]
            if v >= VEL_EPS
                bad_sim = true
                break
            end
        end

        if !bad_sim
            qpos = hcat(qpos, qs[end])
            taus = hcat(taus, pd.τ)
        else
            println("Bad sim with values:")
            @show vs[end]
            println()
        end
    end

    save("./data/gravity_points.jld", "qpos", qpos, "taus", taus)
end

main()
