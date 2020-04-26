using Random

using Blink
using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using StaticArrays

include(joinpath(".", "utils", "utils.jl"))
using .Utils

w = World()

cylinder_bottom_origin = SVector(0., 0., 0.)
cylinder_top_origin = SVector(0., 0., -1.)
r1 = 0.05
r2 = 0.2
r3 = 0.3

add_object!(w, cylinder_bottom_origin, cylinder_top_origin, r1, SVector(0.2, 0.1, 0.4))
add_object!(w, cylinder_bottom_origin, cylinder_top_origin, r2, SVector(0.5, 0.5, 1.))
add_object!(w, cylinder_bottom_origin, cylinder_top_origin, r3, SVector(0.75, -0.5, 0.5))

urdf = joinpath(".", "robot.urdf")
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))

for (i, (low, up)) in enumerate(get_graphics(w))
    setobject!(vis["static_bottom_$i"], low)
    setobject!(vis["static_top_$i"], up)
end

qd = [1.3, 3*pi/4, -2.5]

path = rrt_star(configuration(state), qd, state, w)

@show path



open(vis, Window())

ts = collect(range(1.,length=length(path)))

while true
    animate(vis, ts, path; realtimerate = 1.)
    sleep(length(path))
end
