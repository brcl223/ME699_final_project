using Random

using Blink
using MeshCatMechanisms
using RigidBodyDynamics
using StaticArrays
using MeshCat
using GeometryTypes: Point

include("./utils/kinematics.jl")
using .Kinematics

include("./utils/rrt.jl")
using .RRT

urdf = joinpath(".", "robot.urdf")
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

body = findbody(mechanism, "link_lower_arm")
point = Point3D(default_frame(body), 0., 0., -1.)

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))

open(vis, Window())

circle_origin = SVector(0.5, 0.25, 0.5)
radius = 0.5
ω = 1.0

θ = repeat(range(0, stop=2*pi, length=100), inner=(2,))[2:end]
cx, cy, cz = circle_origin
geometry = PointCloud(Point.(cx .+ radius .* sin.(θ), cy, cz .+ 0.5 .* cos.(θ)))
setobject!(vis[:circle], LineSegments(geometry, LineBasicMaterial()))

function make_circle_controller(state::MechanismState,
                                body::RigidBody,
                                point::Point3D,
                                circle_origin::AbstractVector,
                                radius,
                                ω)
    mechanism = state.mechanism
    world = root_frame(mechanism)
    joint_path = path(mechanism, root_body(mechanism), body)
    point_world = transform(state, point, root_frame(mechanism))
    Jp = point_jacobian(state, joint_path, point_world)
    v̇ = similar(velocity(state))

    function controller!(τ::AbstractVector, t, state)
        desired = Point3D(world, circle_origin .+ radius .* SVector(sin(t/ω), 0, cos(t/ω)))
        point_in_world = transform_to_root(state, body) * point
        point_jacobian!(Jp, state, joint_path, point_in_world)
        Kp = 200
        Kd = 20
        Δp = desired - point_in_world
        v̇ .= Kp * Array(Jp)' * Δp.v .- 20 .* velocity(state)
        τ .= inverse_dynamics(state, v̇)
    end
end

controller! = make_circle_controller(state, body, point, circle_origin, radius, ω)
ts, qs, vs = simulate(state, 10, controller!)

while true
    animate(vis, ts, qs)
    sleep(10)
end
