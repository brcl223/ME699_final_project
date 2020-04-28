using JLD
using RigidBodyDynamics
using StaticArrays

include(joinpath(".", "utils", "utils.jl"))
using .Utils

const NUM_SAMPLES = 10000

function main()
    urdf = joinpath(".", "robot.urdf")
    mechanism = parse_urdf(urdf)
    state = MechanismState(mechanism)
    dr = DynamicsResult(mechanism)
    dim = num_positions(mechanism)

    qpos = zeros(dim,0)
    # Compressed format to store the symmetric
    # matrices in. We'll use this form for the
    # NN anyway
    q̈s = zeros(sum(1:dim),0)

    λ = 10.
    Δt = 1e-5

    for i = 1:NUM_SAMPLES
        if i % 100 == 0
            println("Currently on iteration $i")
        end

        mcs = make_mass_controllers(dim, λ)

        Q̈ = zeros(Float64, (3,3))
        qd = gen_rand_pi(dim)
        for (i, mc) in enumerate(mcs)
            set_configuration!(state, qd)
            zero_velocity!(state)
            ts, _, vs = simulate(state, 2*Δt, mc; Δt=Δt)
            # vs[1] and ts[1] are 0
            # Use the first non-zero state to calculate
            # numerical value of δv/δt
            Q̈[i,:] .= vs[2] ./ ts[2]
        end

        qpos = hcat(qpos, qd)
        mm = λ*inv(Q̈)
        q̈s = hcat(q̈s, sym_to_vec(mm))

        # Q̈ = zeros(Float64, (3,3))
        # qd = gen_rand_pi(dim)
        # for i = 1:dim
        #     set_configuration!(state, qd)
        #     τ = dynamics_bias(state)
        #     τ[i] += λ
        #     dynamics!(dr, state, τ)
        #     Q̈[:,i] .= dr.v̇
        # end

        # # Reset state one more time just in case
        # set_configuration!(state, qd)

    end

    save("./data/mass_points.jld", "qpos", qpos, "qddot", q̈s)
end

main()
