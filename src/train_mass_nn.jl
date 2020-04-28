using Flux, CuArrays
using JLD
using IterTools: ncycle
using LinearAlgebra

using Flux.Data, Flux.Optimise

include("utils/utils.jl")
using .Utils

function main()
    # Seems to be a bug in the Flux package
    # Can't train this on GPU for some reason due to broadcasting
    # Just train on CPU
    nn = build_nn(3, 6)

    opt = ADAM(0.005, (0.9, 0.8))
    data = load("./data/mass_points.jld")
    qpos = data["qpos"]
    qddot = data["qddot"]

    @show size(qpos)
    @show size(qddot)

    train_loader = DataLoader(qpos, qddot, batchsize=256, shuffle=true)
    loss(x,y) = Flux.mse(nn(x), y)
    ps = Flux.params(nn)

    val_qpos = qpos[:,1:100]
    val_qddot = qddot[:,1:100]
    evalcb() = @show(loss(val_qpos, val_qddot))

    train!(loss, ps, ncycle(train_loader, 5), opt, cb=evalcb)
    save_nn(nn, "./models/mass_nn")
end

main()
