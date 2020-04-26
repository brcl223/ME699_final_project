using Flux
using RigidBodyDynamics
using MeshCatMechanisms

include(joinpath(".", "utils", "utils.jl"))
using .Utils

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
        adpd = ADPDController(dim)
        pdgc.qd = pd.qd
        adpd.qd = pd.qd
        final_time = 10.
        ts, qs, vs = simulate(state, final_time, pd; Δt=1e-3)
        tsgc, qsgc, vsgc = simulate(state, final_time, pdgc; Δt=1e-3)
        tsad, qsad, vsad = simulate(state, final_time, adpd; Δt=1e-3)

        println()
        println("Configuration")
        @show qs[end]

        #println()
        #println("Desired Configuration")
        #@show pd.qd

        #println()
        #println("GravComp Configuration")
        #@show qsgc[end]

        println()
        println("Adaptive Configuration")
        @show qsad[end]

        #println()
        #println("Gravity Torques")
        #@show pd.τ

        #println()
        #println("Predicted Torques")
        #pred = pdgc.nn(qs[end])
        #@show pred

        println()
        println("Adaptive Torques")
        @show adpd.τ

        #println()
        #println("MSE Error")
        #@show Flux.mse(pd.τ, pred)

        #println()
        #println("PD Error")
        #@show Flux.mse(pd.qd, qs[end])

        #println()
        #println("PDGC Error")
        #@show Flux.mse(pdgc.qd, qsgc[end])

        println()
        println("ADPD Error")
        @show Flux.mse(adpd.qd, qsad[end])

        animate(mvis, tsad, qsad; realtimerate = 1.)

        readline(stdin)
    end
end



main()
