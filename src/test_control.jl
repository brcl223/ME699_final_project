using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Random
using Blink
using MeshCat
using MeshCatMechanisms
using RigidBodyDynamics
using StaticArrays
using Plots
using DataFrames
using CSV

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
q̇d = velocity(new_state)
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
adpd = ADPDController(q, q̇; Δt=Δt)
pd = PDTracker(q, q̇; Δt=Δt)
#adi = ADPDInertial(q, q̇; Δt=Δt)
zero!(state)

tss, qss, vss = simulate(state, 5., adpd; Δt=Δt)
#tss, qss, vss = simulate(state, 5., adi; Δt=Δt)

zero!(state)
tpd, qpd, vpd = simulate(state, 5., pd; Δt=Δt)

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

MeshCatMechanisms.animate(vis, tss, qss; realtimerate = 1.)
sleep(5)
MeshCatMechanisms.animate(vis, tss, qss; realtimerate = 1.)
sleep(5)
MeshCatMechanisms.animate(vis, tpd, qpd; realtimerate = 1.)
sleep(5)
MeshCatMechanisms.animate(vis, tpd, qpd; realtimerate = 1.)
sleep(5)

cols = length(qd)
rows = length(qss[:,1])
rowspd = length(qpd[:,1])
prows = length(q)

global ess = zeros(rows, cols)
global ėss = zeros(rows, cols)
global epd = zeros(rowspd, cols)
global ėpd = zeros(rowspd, cols)
global q_ss = zeros(rows, cols)
global v_ss = zeros(rows, cols)
global q_pd = zeros(rowspd, cols)
global v_pd = zeros(rowspd, cols)
global qp = zeros(prows, cols)
global vp = zeros(prows, cols)
for ii=1:rows
    q_ss[ii,:] = qss[ii,:][1]
    v_ss[ii,:] = vss[ii,:][1]
    ess[ii,:] = qd - qss[ii,:][1]
    ėss[ii,:] = q̇d - vss[ii,:][1]

    q_pd[ii,:] = qpd[ii,:][1]
    v_pd[ii,:] = vpd[ii,:][1]
    epd[ii,:] = qd - qpd[ii,:][1]
    ėpd[ii,:] = q̇d - vpd[ii,:][1]
end


for ii=1:prows
    qp[ii,:] = q[ii,:][1]
    vp[ii,:] = q̇[ii,:][1]
end

df = DataFrame(A=tss, B=ess[:,1], C=ess[:,2], D=ess[:,3], E=ėss[:,1], F=ėss[:,2], G=ėss[:,3], H=q_ss[:,1], I=q_ss[:,2], J=q_ss[:,3], K=v_ss[:,1], L=v_ss[:,2], M=v_ss[:,3])
CSV.write("adaptSimData.csv", df)

df2 = DataFrame(A=range(0,stop=5,length=length(q)), B=qp[:,1], C=qp[:,2], D=qp[:,3], E=vp[:,1], F=vp[:,2], G=vp[:,3])
CSV.write("pathData.csv", df2)

df3 = DataFrame(A=tpd, B=epd[:,1], C=epd[:,2], D=epd[:,3], E=ėpd[:,1], F=ėpd[:,2], G=ėpd[:,3], H=q_pd[:,1], I=q_pd[:,2], J=q_pd[:,3], K=v_pd[:,1], L=v_pd[:,2], M=v_pd[:,3])
CSV.write("pdSimData.csv", df3)
