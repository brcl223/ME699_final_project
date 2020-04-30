const MAX_NODES = 10000

gen_rand_q() = (rand(3) .- 0.5) .* (4*pi)

mutable struct Node
    q::AbstractArray
    cost::Float64
    parent::Integer
end

Node() = Node(zeros(3), 0., 0)
Node(q) = Node(q, 0., 0)

function steer(qr::AbstractArray,
               qn::AbstractArray,
               val::Float64,
               ϵ::Float64)::AbstractArray
    qnew = qr
    if val >= ϵ
        qnew = qn .+ (qr .- qn) .* ϵ ./ val
    end

    return qnew
end

function is_valid_configuration(q::AbstractArray)::Bool
    # Joints 2 and 3 capped at (-pi,pi)
    @assert length(q) == 3
    if abs(q[1]) > 2*pi
        return false
    elseif abs(q[2]) > pi || abs(q[3]) > pi
        return false
    end
    return true
end

function check_collisions(w::World,
                          lower::RoboCylinder,
                          upper::RoboCylinder,
                          qi::AbstractVector,
                          qf::AbstractVector;
                          steps = 10.)
    d = norm(qf - qi)
    Δd = d / steps
    for i = 0:steps
        q_cur = qi + i * Δd * (qf - qi)
        if check_collisions(w, lower, q_cur) || check_collisions(w, upper, q_cur)
            return true
        end
    end

    return false
end

function rrt_star(initial::AbstractArray,
                  final::AbstractArray,
                  state::MechanismState,
                  w::World;
                  ϵ=1.5,
                  rewire_range=3)
    q_init = copy_segvec(initial)
    q_final = copy_segvec(final)

    # First construct our collisions for the joints
    r = 0.05
    lower_cyl = RoboCylinder(r, state, "link_lower_arm")
    upper_cyl = RoboCylinder(r, state, "link_upper_arm")

    # First check our start and end states to make sure they're valid
    if check_collisions(w, lower_cyl, q_init) ||
        check_collisions(w, upper_cyl, q_init)
        println("Error! Starting configuration is invalid due to collision")
        return nothing
    end

    if check_collisions(w, lower_cyl, q_final) ||
        check_collisions(w, upper_cyl, q_final)
        println("Error: Final configuration is invalid due to collision")
        return nothing
    end

    nodes = Node[]

    push!(nodes, Node(q_init))

    for i = 1:MAX_NODES
        if i % 50 == 0
            println("Currently on iteration $i...")
        end

        # First check if our most recently added node finishes the search
        # There is no need to look through all of them, as they've all been
        # checked up to this point except the most recently added
        if norm(nodes[end].q - q_final) <= ϵ
            push!(nodes, Node(q_final, nodes[end].cost, length(nodes)))
            break
        end

        # If we are at the last iteration and we did not find a node to connect
        # to, then kill the process as it is an invalid path
        if i == MAX_NODES
            println("Error: No solution found in given nodes $MAX_NODES")
            return nothing
        end

        q_rand = gen_rand_q()

        # Find the closest element
        closest = (nothing, Inf, 0)
        for (i, node) in enumerate(nodes)
            dist = norm(node.q - q_rand)
            if dist < closest[2]
                closest = (node, dist, i)
            end
        end

        @assert closest[1] != nothing
        node_near, dist, idx = closest

        # Steer towards the closest element
        q_new = steer(q_rand, node_near.q, dist, ϵ)

        # Check for collision here. If collision continue
        if !is_valid_configuration(q_new) ||
            check_collisions(w, lower_cyl, upper_cyl, node_near.q, q_new)
            continue
        end

        cost = norm(q_new - node_near.q) + node_near.cost

        # Find neighbors in rewiring range
        q_min = node_near.q
        c_min = cost
        idx_min = idx
        for (i, node) in enumerate(nodes)
            # Check collision
            if check_collisions(w, lower_cyl, upper_cyl, node.q, q_new)
                continue
            end
            cur_dist = norm(node.q - q_new)
            cur_cost = cur_dist + node.cost
            if cur_dist <= rewire_range &&
                cur_cost < c_min
                q_min = node.q
                c_min = cur_cost
                idx_min = i
            end
        end

        node_new = Node(q_new, c_min, idx_min)
        push!(nodes, node_new)
    end

    # Finally, build our final path
    node = nodes[end]
    path = typeof(node.q)[]
    while node.parent != 0
        push!(path, node.q)
        node = nodes[node.parent]
    end

    # Push our start node one
    push!(path, first(nodes).q)
    # Now reverse the order
    reverse!(path)
    return path
end
