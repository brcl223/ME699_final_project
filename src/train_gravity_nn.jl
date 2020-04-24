using Flux, CuArrays
using JLD
using IterTools: ncycle
using BSON: @save

using Flux.Data, Flux.Optimise

const LAYER_SIZE = 100

function build_nn(in, out)
    return Chain(
        Dense(in, LAYER_SIZE, relu),
        Dense(LAYER_SIZE, LAYER_SIZE, relu),
        Dense(LAYER_SIZE, LAYER_SIZE, relu),
        Dense(LAYER_SIZE, out),
    )
end

function main()
    nn = build_nn(3, 3) |> gpu

    opt = ADAM(0.005, (0.9, 0.8))
    data = load("./data/gravity_points.jld")
    qpos = data["qpos"] |> gpu
    taus = data["taus"] |> gpu

    train_loader = DataLoader(qpos, taus, batchsize=256, shuffle=true)
    loss(x,y) = Flux.mse(nn(x), y)
    ps = Flux.params(nn)

    val_qpos = qpos[:,1:100]
    val_taus = taus[:,1:100]
    evalcb() = @show(loss(val_qpos, val_taus))

    train!(loss, ps, ncycle(train_loader, 20), opt, cb=evalcb)

    # Models saved while on GPU may not load correctly
    # Make sure to save while on CPU
    nn = cpu(nn)
    @save "./models/gravity_nn.bson" nn
end

main()
