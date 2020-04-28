const LAYER_SIZE = 100

function build_nn(in::Integer, out::Integer)
    return Chain(
        Dense(in, LAYER_SIZE, relu),
        Dense(LAYER_SIZE, LAYER_SIZE, relu),
        Dense(LAYER_SIZE, LAYER_SIZE, relu),
        Dense(LAYER_SIZE, out),
    )
end

# Default saving model as CPU model to avoid loading issues
function save_nn(nn::Chain, filename)
    nn = cpu(nn)
    @save "$(filename).bson" nn
end

# Default loading model onto GPU
function load_nn(filename; use_gpu=false)
    @load "$(filename).bson" nn
    if use_gpu
        nn = gpu(nn)
    end
    return nn
end
