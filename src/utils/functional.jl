mutable struct Quaternion
    w::Float64
    x::Float64
    y::Float64
    z::Float64
end

mutable struct EulerAngles
    roll::Float64
    pitch::Float64
    yaw::Float64
end

function to_euler_angles(q::Quaternion)
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    roll = atan(sinr_cosp, cosr_cosp)

    sinp = 2 * (q.w * q.y - q.z * q.x)
    pitch = 0
    if abs(sinp) >= 1
        pitch = sign(sinp) * pi / 2.0
    else
        pitch = asin(sinp)
    end

    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = atan(siny_cosp, cosy_cosp)

    return EulerAngles(roll, pitch, yaw)
end

function bound_2pi(v::AbstractVector)
    bound = floor.(v ./ (2*pi))
    return v .- 2*pi .* bound
end
