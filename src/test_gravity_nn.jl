using Flux
using RigidBodyDynamics
using MeshCatMechanisms

include("./utils/controllers.jl")
using .Controllers

function main()
    urdf = joinpath(".", "robot.urdf")
    mechanism = parse_urdf(urdf)
    state = MechanismState(mechanism)
    dim = num_positions(mechanism)
    pdgc = PDGCController(dim)
    mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
    open(mvis)

    while true
        pd = PDController(dim)
        pdgc.qd = pd.qd
        final_time = 10.
        ts, qs, vs = simulate(state, final_time, pd; Δt=1e-3)
        tsgc, qsgc, vsgc = simulate(state, final_time, pdgc; Δt=1e-3)

        println("Configuration")
        @show qs[end]

        println("Desired Configuration")
        @show pd.qd

        println("GravComp Configuration")
        @show qsgc[end]

        println("Gravity Torques")
        @show pd.τ

        println("Predicted Torques")
        pred = pdgc.nn(qs[end])
        @show pred

        println("MSE Error")
        @show Flux.mse(pd.τ, pred)

        println("PD Error")
        @show Flux.mse(pd.qd, qs[end])

        println("PDGC Error")
        @show Flux.mse(pdgc.qd, qsgc[end])

        animate(mvis, ts, qs; realtimerate = 1.)

        readline(stdin)
    end
end



main()
