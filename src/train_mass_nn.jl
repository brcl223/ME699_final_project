using Flux, CuArrays
using JLD
using IterTools: ncycle
using LinearAlgebra

using Flux.Data, Flux.Optimise
using Flux: Params

using DataFrames
using CSV

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

    q_train = qpos[:,201:end]
    qddot_train = qddot[:,201:end]

    q_test = qpos[:,1:200]
    qddot_test = qddot[:,1:200]

    @show size(q_train)
    @show size(qddot_train)

    train_loader = DataLoader(q_train, qddot_train, batchsize=256, shuffle=true)
    loss(x,y) = Flux.mse(nn(x), y)
    ps = Flux.params(nn)

    val_qpos = q_train[:,1:100]
    val_qddot = qddot_train[:,1:100]

    # Attribution: Flux documentation, Custom Training Loop
    function custom_train!(loss, ps, data, opt)
        tloss = []
        vloss = []
        dloss = []

        i = 1

        ps = Params(ps)
        for d in data
            println("Iteration $i")
            i += 1
            gs = gradient(ps) do
                training_loss = loss(d...)
                return training_loss
            end

            update!(opt, ps, gs)
            push!(dloss, loss(d...))
            push!(tloss, loss(q_test, qddot_test))
            push!(vloss, loss(val_qpos, val_qddot))
        end

        return (tloss, vloss, dloss)
    end

    # Collect our loss metrics
    tloss, vloss, dloss = custom_train!(loss, ps, ncycle(train_loader, 5), opt)
    df = DataFrame(A=tloss, B=vloss, C=dloss)
    CSV.write("./data/mass_nn_loss.csv", df)

    # Finally, save our neural network
    save_nn(nn, "./models/mass_nn")
end

main()
