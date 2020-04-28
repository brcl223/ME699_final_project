##############################################################
# Demonstration.jl
#
# Program written by:
# Benton Clark
# Brian Moberly
# Ethan Howell
#
# This is a 3DOF 3R robot simulation. This demonstration
# will consist of 11 phases:
#
# 1) Build a world with various collidable obstacles
# 2) Place object of unknown mass into world at start state
# 3) Have "Camera" with noisy sensor readings find
#    object in its coordinate frame. Reading will be
#    filtered with a Kalman Filter, and inverse
#    kinematics will be used to map this 3D Cartesian
#    point into the robots joint space
# 4) RRT* will be used to find way-points between the
#    initial and final state for the object to pick up the
#    mass
# 5) A trajectory will be plotted, providing a list of
#    desired (q,q̇,q̈) for each point in the trajectory
# 6) A tracking controller is used in combination with a
#    neural network predicting the mass matrix of the robot
#    for each time step. Noisy sensor data will be provided
#    as the actual (q,q̇,q̈) values, and a Kalman filter will
#    be used to clean these up
# 7) Once the robot has reached the object to acquire, it
#    will "pick it up", adding mass to the end effector
# 8) A new trajectory will be plotted as before to guide
#    the robot collision free to the goal state to drop
#    the object
# 9) An adaptive controller with the neural net mass
#    estimation will be used to track the final trajectory
# 10) A new trajectory will be plotted back to the original
#     starting configuration (0ᵥ)
# 11) Lastly, the robot will return to its initial
#     configuration with a trajectory tracking controller
##############################################################

using Blink
using ColorTypes
using CoordinateTransformations
using Distributions
using Flux
using GeometryTypes
using LinearAlgebra
using MeshCat
using MeshCatMechanisms
using Random
using RigidBodyDynamics
using StaticArrays

const D = Distributions
const RGD = RigidBodyDynamics

include(joinpath("utils", "utils.jl"))
using .Utils

##############################################################
# Configuration: Parameters for program
#
# Note: Mass of object can be modified, but to do so, the
# mass and Ixx,Iyy values must be updated in robot_heavy.urdf
##############################################################

# Variance of noise for Kalman filter
variance = 0.1
noise = D.Normal(0, variance)

# Delta time for simulation accuracy
Δt = 1e-3
# Mass Neural Network for Controller
nn = load_nn("models/mass_nn")

# State matrices for Kalman filter used for object identification
F(_) = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
G(_) = zeros(3, 3)
H = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
σ = ones(3) * var(noise)
P₀ = Diagonal(var(noise)*ones(3,3))


# State matrices for Kalman Filter used for tracking
Fₜ(Δt) = [1 Δt  0  0  0  0;
         0  1  0  0  0  0;
         0  0  1 Δt  0  0;
         0  0  0  1  0  0;
         0  0  0  0  1 Δt;
         0  0  0  0  0  1]
Gₜ(Δt) = [0.5*Δt^2 0      0;
         Δt       0      0;
         0   0.5*Δt^2    0;
         0       Δt      0;
         0       0   0.5Δt^2;
         0       0      Δt]
Hₜ = diagm(ones(6))
σₜ = ones(6) * var(noise)
Pₜ₀ = Diagonal(var(noise)*ones(6,6))

# Start platform for object
hsₚ = 0.2
sᵢ = SVector(1.5, 0, -0.2)
# Start state for the object
rₒ = 0.1
pₒ = SVector(sᵢ[1], sᵢ[2], hsₚ + sᵢ[3] + rₒ)
# Goal platform for object
hgₚ = 0.2
sg = SVector(-0.75, -0.75, 0.5)
# Goal state for object
pₑ = SVector(sg[1], sg[2], sg[3] + hgₚ + rₒ)






##############################################################
# Phase 0: Load non-mass robot model
##############################################################
urdf = joinpath(".", "robot.urdf")
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)






##############################################################
# Phase 1: Build world
##############################################################
printstyled("PHASE 1: Building World...\n"; bold=true, color=:magenta)

w = World()

# Tuple of (radius, height, location) for collision cylinders
rh = [(0.1, 0.1, SVector(0.4, 0.4, -0.3)),
      (0.05, 0.6, SVector(-0.5, -1., 0.)),
      (0.05, 0.1, SVector(0.7, -0.8, 0.75)),
      (0.125, 0.75, SVector(-0.25, 0.25, -0.3)),
      (0.25, hsₚ, sᵢ), # Start state
      (0.25, hgₚ, sg)] # Goal state

for (r, h, loc) in rh
    add_object!(w, r, h, loc)
end




##############################################################
# Phase 2: Place ball of unknown mass in world
##############################################################
printstyled("PHASE 2: Placing object of unknown mass...\n"; bold=true, color=:magenta)

# Maybe delete this phase since not using it now?
# Ball and platform params are setup above in configuration




##############################################################
# Phase 3: Find location of ball in joint space
##############################################################
printstyled("PHASE 3: Finding object location...\n"; bold=true, color=:magenta)

# Get initial guess for object state p̂ₒ from 10 "camera" readings
p̂₀ = mean(rand(noise, 3, 10) .+ pₒ; dims=2)[:,1]
kf = KalmanFilter(F,G,H,σ,p̂₀,P₀)

# Update filter on 10000 readings to get an accurate estimate
for i = 1:10000
    sample = pₒ + rand(noise, 3)
    update_filter!(kf, sample, zeros(3))
end

p̂ₙ, _ = update_filter!(kf, pₒ + rand(noise, 3), zeros(3))

set_configuration!(state, [0, pi, 0])
xd = Point3D(root_frame(mechanism), p̂ₙ)
lower_arm = findbody(mechanism, "link_lower_arm")
ee = Point3D(default_frame(lower_arm), 0., 0., -1.)

new_state = jacobian_transpose_ik!(state, lower_arm, ee, xd)

qd = copy(configuration(new_state))
zero!(state)








##############################################################
# Phase 4: RRT* Waypoint Planning
##############################################################
printstyled("PHASE 4: Building waypoint tree...\n"; bold=true, color=:magenta)

path = rrt_star(configuration(state), qd, state, w)








##############################################################
# Phase 5: Path trajectory through waypoints
##############################################################
printstyled("PHASE 5: Building trajectory from waypoints...\n"; bold=true, color=:magenta)

traj_qᵢ, traj_q̇ᵢ = plan_trajectory(path)









##############################################################
# Phase 6: Tracking the trajectory with noisy controller
##############################################################
printstyled("PHASE 6: Tracking trajectory to object...\n"; bold=true, color=:magenta)

x̂₀ = zeros(6)
kfs = KalmanFilterState(Fₜ,Gₜ,Hₜ,σₜ,x̂₀,Pₜ₀)
add_state!(kfs, "accel", zeros(3))

pdᵢ = PDTracker(traj_qᵢ, traj_q̇ᵢ, nn, kfs; Δt=Δt)
# pd = ADPDController(q,q̇)
zero!(state)

tssᵢ, qssᵢ, vssᵢ = simulate(state, 10., pdᵢ; Δt=Δt)




##############################################################
# Phase 7: "Picking up object" (Switching robot model)
##############################################################
printstyled("PHASE 7: Picking up object...\n"; bold=true, color=:magenta)

urdf_heavy = joinpath(".", "robot_heavy.urdf")
mechanism_heavy = parse_urdf(urdf_heavy)
state_heavy = MechanismState(mechanism_heavy)





##############################################################
# Phase 8: Plot new trajectory
##############################################################
printstyled("PHASE 8: Building trajectory to goal...\n"; bold=true, color=:magenta)

qᵢ = copy(qssᵢ[end])

# Use Kalman filter again to estimate location of goal state
# Get initial guess for object state p̂ₑ from 10 "camera" readings
p̂ₑ = mean(rand(noise, 3, 10) .+ pₑ; dims=2)[:,1]
kf = KalmanFilter(F,G,H,σ,p̂ₑ,P₀)

# Update filter on 10000 readings to get an accurate estimate
for i = 1:10000
    sample = pₑ + rand(noise, 3)
    update_filter!(kf, sample, zeros(3))
end

p̂ₑ, _ = update_filter!(kf, pₑ + rand(noise, 3), zeros(3))

# Find our configuration in joint space
# This initial configuration increases likelihood of arm not
# being through platform
set_configuration!(state, [pi, pi, 0])
xg = Point3D(root_frame(mechanism), p̂ₑ)

goal_state = jacobian_transpose_ik!(state, lower_arm, ee, xg)
qg = copy(configuration(goal_state))

set_configuration!(state, qᵢ)
zero_velocity!(state)
path_goal = rrt_star(qᵢ, qg, state, w)

traj_qg, traj_q̇g = plan_trajectory(path_goal)







##############################################################
# Phase 9: Adaptive control with mass held
##############################################################
printstyled("PHASE 9: Tracking trajectory to goal...\n"; bold=true, color=:magenta)

# Here we use our heavy robot

set_configuration!(state_heavy, qᵢ)
zero_velocity!(state_heavy)
x̂₀ = combine_joint_state(configuration(state_heavy), velocity(state_heavy))
kfs = KalmanFilterState(Fₜ,Gₜ,Hₜ,σₜ,x̂₀,Pₜ₀)
add_state!(kfs, "accel", zeros(3))

# pdg = PDTracker(traj_qg, traj_q̇g, nn, kfs; Δt=Δt)
pdg = ADPDController(traj_qg, traj_q̇g, nn, kfs; Δt=Δt)

tssg, qssg, vssg = simulate(state_heavy, 10., pdg; Δt=Δt)







##############################################################
# Phase 10: Plot trajectory back to starting position
##############################################################
printstyled("PHASE 10: Plotting trajectory home...\n"; bold=true, color=:magenta)

set_configuration!(state, qg)
zero_velocity!(state)
path_home = rrt_star(qg, zeros(3), state, w)
traj_qhome, traj_q̇home = plan_trajectory(path_home)





##############################################################
# Phase 11: Track trajectory back to starting position
##############################################################
printstyled("PHASE 11: I'm sorry Dave, I'm afraid I must go home...\n"; bold=true, color=:magenta)

set_configuration!(state, qg)
zero_velocity!(state)
x̂₀ = combine_joint_state(configuration(state), velocity(state))
kfs = KalmanFilterState(Fₜ,Gₜ,Hₜ,σₜ,x̂₀,Pₜ₀)
add_state!(kfs, "accel", zeros(3))

pdhome = PDTracker(traj_qhome, traj_q̇home, nn, kfs; Δt=Δt)

tsshome, qsshome, vsshome = simulate(state, 10., pdhome; Δt=Δt)







##############################################################
# Collect data from trajectories
##############################################################

# Collect data here





##############################################################
# Plot the trajectories and run the animations
##############################################################

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
x_traj_qᵢ = SVector{3,Float64}[]
x_traj_qg = SVector{3,Float64}[]
x_traj_qhome = SVector{3,Float64}[]

for c in traj_qᵢ
    set_configuration!(state, c)
    push!(x_traj_qᵢ, RGD.transform(state, ee, root_frame(mechanism)).v)
end

for c in traj_qg
    set_configuration!(state, c)
    push!(x_traj_qg, RGD.transform(state, ee, root_frame(mechanism)).v)
end

for c in traj_qhome
    set_configuration!(state, c)
    push!(x_traj_qhome, RGD.transform(state, ee, root_frame(mechanism)).v)
end

pc_traj_qᵢ = PointCloud(x_traj_qᵢ)
ls_traj_qᵢ = LineSegments(pc_traj_qᵢ, LineBasicMaterial(color=RGBA(0,1,0,1)))

pc_traj_qg = PointCloud(x_traj_qg)
ls_traj_qg = LineSegments(pc_traj_qg, LineBasicMaterial(color=RGBA(0,1,0,1)))

pc_traj_qhome = PointCloud(x_traj_qhome)
ls_traj_qhome = LineSegments(pc_traj_qhome, LineBasicMaterial(color=RGBA(0,1,0,1)))


x_qssᵢ = SVector{3,Float64}[]
x_qssg = SVector{3,Float64}[]
x_qsshome = SVector{3,Float64}[]

for c in qssᵢ
    set_configuration!(state, c)
    push!(x_qssᵢ, RGD.transform(state, ee, root_frame(mechanism)).v)
end

for c in qssg
    set_configuration!(state, c)
    push!(x_qssg, RGD.transform(state, ee, root_frame(mechanism)).v)
end

for c in qsshome
    set_configuration!(state, c)
    push!(x_qsshome, RGD.transform(state, ee, root_frame(mechanism)).v)
end

pc_qssᵢ = PointCloud(x_qssᵢ)
pc_qssg = PointCloud(x_qssg)
pc_qsshome = PointCloud(x_qsshome)

ls_qssᵢ = LineSegments(pc_qssᵢ, LineBasicMaterial(color=RGBA(1,0,0,1)))
ls_qssg = LineSegments(pc_qssg, LineBasicMaterial(color=RGBA(1,0,0,1)))
ls_qsshome = LineSegments(pc_qsshome, LineBasicMaterial(color=RGBA(1,0,0,1)))

object = HyperSphere(Point(0.,0.,0.), rₒ)

for (i, (low, up)) in enumerate(get_graphics(w))
    setobject!(vis["static_bottom_$i"], low)
    setobject!(vis["static_top_$i"], up)
end

colors = [RGBA(1,1,1,1),
          RGBA(1,1,1,1),
          RGBA(1,1,1,1),
          RGBA(1,1,1,1),
          RGBA(1,0,0,1),
          RGBA(0,1,0,1)]

for (i, (c, color)) in enumerate(zip(get_cylinders(w), colors))
    setobject!(vis["cylinder"]["$i"], c, MeshPhongMaterial(color=color))
end

open(vis, Window())


# Build ball animation
function build_anim(vis, tssg, x_qssg)
    current_frame::Integer = -1
    ball_anim = Animation()
    FPS = 30
    for i = 1:length(tssg)
        if current_frame < floor(tssg[i] * FPS)
            current_frame += 1
        else
            continue
        end

        atframe(ball_anim, current_frame) do
            settransform!(vis["mass_object"], Translation(x_qssg[i]))
        end
    end

    return ball_anim
end

ball_anim = build_anim(vis, tssg, x_qssg)


# Finally run the animation with all trajectories plotted
zero!(state)
set_configuration!(vis, configuration(state))
setobject!(vis["mass_object"], object, MeshPhongMaterial(color=RGBA(0,0,1,1)))
settransform!(vis["mass_object"], Translation(pₒ))
sleep(3)
while true
    zero!(state)
    set_configuration!(vis, configuration(state))
    setobject!(vis["traj"], ls_traj_qᵢ)
    setobject!(vis["qss"], ls_qssᵢ)
    settransform!(vis["mass_object"], Translation(pₒ))
    animate(vis, tssᵢ, qssᵢ; realtimerate = 1.)
    sleep(5)
    set_configuration!(state, qᵢ)
    set_configuration!(vis, qᵢ)
    setobject!(vis["traj"], ls_traj_qg)
    setobject!(vis["qss"], ls_qssg)
    setanimation!(vis, ball_anim)
    animate(vis, tssg, qssg; realtimerate = 1.)
    sleep(5)
    set_configuration!(state, qg)
    set_configuration!(vis, qg)
    setobject!(vis["traj"], ls_traj_qhome)
    setobject!(vis["qss"], ls_qsshome)
    settransform!(vis["mass_object"], Translation(pₑ))
    animate(vis, tsshome, qsshome; realtimerate = 1.)
    sleep(5)
end
