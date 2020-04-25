using Blink
using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using StaticArrays
using GeometryTypes
using ColorTypes
using Random

urdf = joinpath(".", "robot.urdf")
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

body = findbody(mechanism, "link_lower_arm")
point = Point3D(default_frame(body), 0., 0., -0.5)
widths = Vec(@SVector [0.1, 0.1, 1])
origin = Vec(@SVector [-0.05, -0.05, -1])
# rect = HyperRectangle(point, widths)
rect = HyperRectangle(origin, widths)

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
setelement!(vis, point.frame, rect)

open(vis, Window())

while true
    @show transform_to_root(state, body)
    sleep(2)
    rand!(state)
    set_configuration!(vis, configuration(state))
end
