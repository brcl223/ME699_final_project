import Pkg;
# Pkg.activate(@__DIR__);
using RigidBodyDynamics, RigidBodySim, DifferentialEquations
using LinearAlgebra
using StaticArrays
#using Plots
using MeshCat, MeshCatMechanisms

include("./PandaRobot.jl")

panda = PandaRobot.Panda()
mechanism = panda.mechanism
state = MechanismState(mechanism)
set_configuration!(state, [0.0; 0.0; 0.0; 0.0; 0.0; pi/2.0; 0.0])
zero_velocity!(state)

function control!(τ, t, state)
    # Do some PD
    τ .= -20 .* velocity(state) - 100*(configuration(state) - [0.0; 0.0; 0.0; pi/2.0; 0.0; pi/2.0; 0.0; 0.0; 0.0])
end

final_time = 10.
ts, qs, vs = simulate(state, final_time, control!; Δt=1e-3)
mvis = MechanismVisualizer(mechanism, URDFVisuals(PandaRobot.urdfpath()))
open(mvis)
animate(mvis, ts, qs; realtimerate=1.)

while true
    sleep(15)
    animate(mvis, ts, qs; realtimerate=1.)
end
