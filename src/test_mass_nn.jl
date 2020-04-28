using Flux
using RigidBodyDynamics

include("utils/utils.jl")
using .Utils

function main()
    urdf = joinpath(".", "robot.urdf")
    mechanism = parse_urdf(urdf)
    state = MechanismState(mechanism)
    nn = load_nn("models/mass_nn")
    dim = num_positions(mechanism)

    while true
        q = gen_rand_pi(dim)
        println()
        println("Configuration")
        display(q)

        println()
        println("NN")
        display(vec_to_sym(nn(q)))

        set_configuration!(state, q)
        println()
        println("M(q)")
        display(mass_matrix(state))

        readline(stdin)
    end
end

main()
