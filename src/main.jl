using RigidBodyDynamics
using MeshCatMechanisms

urdf = joinpath(".", "robot.urdf")
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

@show mechanism
@show configuration(state)

function control!(τ::AbstractVector, t, state::MechanismState)
    desired_state = [pi/2.; pi/2.; pi/2.]
    τ .= -20 .* (configuration(state) - desired_state) - 10 .* velocity(state)
end

function no_control!(τ, t, state)
    τ = [0;0;0]
end

mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
open(mvis)

set_configuration!(state, [pi/2.; pi/2.; pi/2.])
final_time = 10.
ts, qs, vs = simulate(state, final_time, no_control!; Δt = 1e-3)

while true
    animate(mvis, ts, qs; realtimerate = 1.)
    sleep(5)
end
