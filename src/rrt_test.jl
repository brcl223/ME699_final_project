using Random

using Blink
using MeshCatMechanisms
using RigidBodyDynamics
using StaticArrays

include("./utils/rrt.jl")
using .RRT

urdf = joinpath(".", "robot.urdf")
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

qd = [1.3, 3*pi/4, -2.5]

path = rrt_star(configuration(state), qd)

@show path

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
open(vis, Window())

ts = collect(range(1.,length=length(path)))

sleep(5)

animate(vis, ts, path; realtimerate = 1.)

while true
    sleep(5)
    animate(vis, ts, path; realtimerate = 1.)
end
