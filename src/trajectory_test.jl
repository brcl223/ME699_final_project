using Blink
using LinearAlgebra
using StaticArrays
using MeshCat
using GeometryTypes
using CoordinateTransformations

include("./utils/trajectories.jl")

x = SVector{3,Float64}[]

push!(x, @SVector(zeros(3)))
push!(x, @SVector(ones(3)))
push!(x, @SVector([1.1, 2.2, 0.5]))
push!(x, @SVector([1.5, 3.2, -3.0]))
push!(x, @SVector([1.9, -2.2, 4.3]))

q, q̇ = plan_trajectory(x)

@show length(q)
@show length(q̇)
@show q[end-10:end]

pc = PointCloud(q)
ls = LineSegments(pc, LineBasicMaterial())

vis = Visualizer()
open(vis, Window())

for (i, y) in enumerate(x)
    setobject!(vis["sphere_$i"], HyperSphere(Point3f0(0), 0.1f0))
    settransform!(vis["sphere_$i"], Translation(y))
end

setobject!(vis, ls)

while true
    sleep(1)
end
