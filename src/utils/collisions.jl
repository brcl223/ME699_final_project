to_affine_map(tform::Transform3D) = CT.AffineMap(rotation(tform), translation(tform))
to_affine_map(tform::Transform3D, X::LazySet{N}) where N = LS.AffineMap(rotation(tform), X, translation(tform))
to_affine_map(trans::AbstractVector, X::LazySet{N}) where N = LS.AffineMap(I, X, trans)

function to_transform_3d(v::AbstractVector, m::Mechanism)
    world = root_frame(mechanism)
    return Transform3D(world, world, v)
end

mutable struct Cylinder{N}
    to::AbstractVector
    bo::AbstractVector
    r::Float64
    top_points::AbstractVector
    bottom_points::AbstractVector
    P::LazySet{N}
end

function Cylinder(r::Float64, h::Float64 = -1.)
    bo = SVector(0., 0., 0.)
    to = SVector(0., 0., h)
    θ = repeat(range(0, stop=2*pi, length=10), inner=(2,))[2:end]
    bx, by, bz = bo
    tx, ty, tz = to

    bottom_circle = Point.(bx .+ r .* cos.(θ), by .+ r .* sin.(θ), bz)
    top_circle = Point.(tx .+ r .* cos.(θ), ty .+ r .* sin.(θ), tz)

    all_points = vcat(bottom_circle, top_circle)
    P = VPolytope(all_points)
    return Cylinder(to, bo, r, top_circle, bottom_circle, P)
end

function graphics_transform(c::Cylinder, af::CT.AbstractAffineMap)
    bottom_rotated = c.bottom_points |> point -> map(p -> af(p), point)
    top_rotated = c.top_points |> point -> map(p -> af(p), point)
    geometry_bottom = PointCloud(bottom_rotated)
    geometry_top = PointCloud(top_rotated)

    lsb = LineSegments(geometry_bottom, LineBasicMaterial())
    lst = LineSegments(geometry_top, LineBasicMaterial())

    return (lsb, lst)
end

function graphics_transform(c::Cylinder, v::AbstractVector)
    af = CT.Translation(v)
    return graphics_transform(c, af)
end

function graphics_transform(c::Cylinder, tform::Transform3D)
    af = to_affine_map(tform)
    return graphics_transform(c, af)
end

function collision_transform(c::Cylinder, tform::Transform3D)
    return to_affine_map(tform, c.P)
end

function collision_transform(c::Cylinder, trans::AbstractVector)
    return to_affine_map(trans, c.P)
end

function apply_transform(c::Cylinder, tform::Transform3D)
    af = to_affine_map(tform)
    lsb, lst = graphics_transform(c, af)
    afP = collision_transform(c, tform)
    return afP, lsb, lst
end

# Cylinder specfically for Robot
mutable struct RoboCylinder{N}
    c::Cylinder{N}
    state::MechanismState
    body::RigidBody
end

function RoboCylinder(r, state, body_name)
    c = Cylinder(r)
    body = findbody(state.mechanism, body_name)
    return RoboCylinder(c, state, body)
end

function check_collision!(rc::RoboCylinder,
                          q::AbstractVector,
                          other::LS.AbstractAffineMap)::Bool
    set_configuration!(rc.state, q)
    tform = transform_to_root(rc.state, rc.body)
    afP = collision_transform(rc.c, tform)
    dir = @SVector(rand(3)) .- 0.5
    return collision_detection(afP, other, dir)
end

function get_graphics(rc::RoboCylinder, q::AbstractVector)
    set_configuration!(rc.state, q)
    tform = transform_to_root(rc.state, rc.body)
    return graphics_transform(rc.c, tform)
end

mutable struct World
    cylinders
    locations
    graphics
end

function World()
    World([], [], [])
end

function add_object!(w::World,
                     r::Float64,
                     h::Float64,
                     loc::AbstractVector)
    c = Cylinder(r, h)
    push!(w.cylinders, c)
    push!(w.locations, loc)
    push!(w.graphics, graphics_transform(c, loc))
end

function get_graphics(w::World)
    return w.graphics
end

function get_cylinders(w::World)
    out = []
    for (c, loc) in zip(w.cylinders, w.locations)
        af = CT.Translation(loc)
        c1 = af(c.to)
        c2 = af(c.bo)
        cylinder = GT.Cylinder(Point(c1), Point(c2), c.r)
        push!(out, cylinder)
    end

    return out
end

function check_collisions(w::World, rc::RoboCylinder, q::AbstractVector)::Bool
    for (i, c) in enumerate(w.cylinders)
        loc = w.locations[i]
        afC = collision_transform(c, loc)
        if check_collision!(rc, q, afC)
            return true
        end
    end

    return false
end


mutable struct Sphere{N}
    P::LazySet{N}
end
