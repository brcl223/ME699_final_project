using Random

using Blink
using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using StaticArrays

include(joinpath(".", "utils", "utils.jl"))
using .Utils

w = World()

cbo = SVector(0., 0., 0.)
cto = SVector(0., 0., -1.)
r = 0.1

add_object!(w, cbo, cto, r, SVector(0.4, 0.4, 0.5))

urdf = joinpath(".", "robot.urdf")
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

# First target is in Task space. Convert to joint space
# to run RRT
xd = Point3D(root_frame(mechanism), 1., 0.5, -0.7)
body = findbody(mechanism, "link_lower_arm")
point = Point3D(default_frame(body), 0., 0., -1.)

new_state = jacobian_transpose_ik!(state, body, point, xd)

@show configuration(new_state)
@show transform(new_state, point, root_frame(mechanism))

qd = copy(configuration(new_state))
zero!(state)

path = rrt_star(configuration(state), qd, state, w)

@show path

q, q̇ = plan_trajectory(path)
x = SVector{3,Float64}[]

for c in q
    set_configuration!(state, c)
    push!(x, transform(state, point, root_frame(mechanism)).v)
end

ts = Float64[]
for t = 1:length(q)
    push!(ts, (t-1)*1e-3)
end

@show length(q)
@show length(q̇)
@show length(ts)
@show q[end-10:end]

Δt = 1e-3
pd = PDTracker(q, q̇; Δt=Δt)
zero!(state)

tss, qss, vss = simulate(state, 5., pd; Δt=Δt)

@show length(tss)
@show length(qss)

pc = PointCloud(x)
ls = LineSegments(pc, LineBasicMaterial())

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))

for (i, (low, up)) in enumerate(get_graphics(w))
    setobject!(vis["static_bottom_$i"], low)
    setobject!(vis["static_top_$i"], up)
end

open(vis, Window())

setobject!(vis["traj"], ls)

while true
    animate(vis, tss, qss; realtimerate = 1.)
    sleep(5)
end
