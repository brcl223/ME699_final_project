# Trajectory Planning algorithm modified from "Robot Manipulator Control Theory and Practice",
# Section 4.2 "Path Generation"
function plan_trajectory(qs::AbstractVector{<:AbstractVector{T}};
                         q̇_max=1, Δt=1e-3, ϵ=1e-4, max_step_size=20) where T
    # Initial and final velocities should be 0
    # Others may need tweaking

    N = length(qs[1])
    points = SVector{N,Float64}[]
    velocities = SVector{N,Float64}[]

    push!(points, copy(qs[1]))
    push!(velocities, @SVector(zeros(N)))

    d_max = ϵ * max_step_size

    # We will use continuous acceleration until a q̇ max is reached
    current_step = 1
    for i = 2:length(qs) - 1
        q_init = points[end]
        q_cur = q_init
        q_goal = qs[i]
        d = norm(q_init - q_goal)
        # Δd represents the number of discrete distances between
        # the two points
        Δd = floor(d / ϵ)

        println("Number of steps between points: $(Δd)")

        steps_taken = 1
        while Δd > steps_taken + current_step
            q_next = q_init + (ϵ * steps_taken / d) .* (q_goal - q_init)
            q̇_next = (q_next - q_cur) ./ Δt
            push!(points, q_next)
            push!(velocities, q̇_next)
            steps_taken += current_step
            current_step = min(current_step + 1, max_step_size)
            q_cur = q_next
        end

        # Add last point
        q̇_goal = (q_goal - q_cur) / Δt
        push!(points, copy(q_goal))
        push!(velocities, q̇_goal)
    end

    # For the last point we have to make sure we slow down just as
    # we initially sped up
    q_init = points[end]
    q_cur = q_init
    q_goal = qs[end]
    d = norm(q_init - q_goal)
    Δd = floor(d / ϵ)
    steps_to_stop = sum(1:max_step_size)
    steps_taken = 0

    println("Steps on last run: $Δd")

    if Δd < steps_to_stop
        error("ERROR: Not enough steps to slow down")
    end

    while Δd - steps_taken > (current_step + max_step_size)
        q_next = q_init + (ϵ * steps_taken / d) .* (q_goal - q_init)
        q̇_next = (q_next - q_cur) ./ Δt
        push!(points, q_next)
        push!(velocities, q̇_next)
        q_cur = q_next
        steps_taken += current_step
        current_step = min(current_step + 1, max_step_size)
    end

    while Δd > steps_taken + 1
        q_next = q_init + (ϵ * steps_taken / d) .* (q_goal - q_init)
        q̇_next = (q_next - q_cur) ./ Δt
        push!(points, q_next)
        push!(velocities, q̇_next)
        q_cur = q_next
        steps_taken += current_step
        current_step = max(current_step - 1, 1)
    end

    # Finally, add last point
    push!(points, copy(qs[end]))
    push!(velocities, @SVector(zeros(N)))

    return (points, velocities)
end
