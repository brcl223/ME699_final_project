using Blink
using LazySets
using StaticArrays
using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using GeometryTypes
using CoordinateTransformations
const CT = CoordinateTransformations
const LS = LazySets
using Random
using LinearAlgebra
using ConvexBodyProximityQueries

include("./utils/collisions.jl")
using .Collisions

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
lower_body = findbody(mechanism, "link_lower_arm")
upper_body = findbody(mechanism, "link_upper_arm")
world = root_frame(mechanism)

lower_cyl = RoboCylinder(cylinder_bottom_origin, cylinder_top_origin, 0.1, state, "link_lower_arm")
upper_cyl = RoboCylinder(cylinder_bottom_origin, cylinder_top_origin, 0.1, state, "link_upper_arm")

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
set_configuration!(vis, configuration(state))

for (i, (low, up)) in enumerate(get_graphics(w))
    setobject!(vis["static_bottom_$i"], low)
    setobject!(vis["static_top_$i"], up)
end

open(vis, Window())

while true
    rand!(state)
    q = configuration(state)

    if check_collisions(w, lower_cyl, q)
        println("Collision with Lower Cylinder")
    end

    if check_collisions(w, upper_cyl, q)
        println("Collision with Upper Cylinder")
    end
   
    bottom_ml, top_ml = get_graphics(lower_cyl, q)
    bottom_mu, top_mu = get_graphics(upper_cyl, q)

    setobject!(vis[:moving_bottom_l], bottom_ml)
    setobject!(vis[:moving_top_l], top_ml)
    setobject!(vis[:moving_bottom_u], bottom_mu)
    setobject!(vis[:moving_top_u], top_mu)
    set_configuration!(vis, configuration(state))
    sleep(3)
end
