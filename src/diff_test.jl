using ForwardDiff
using Flux

include("utils/utils.jl")
using .Utils

nn = load_nn("models/mass_nn")

J = ForwardDiff.jacobian(nn, [1.,2.,3.])

δMq1 = vec_to_sym(J[:,1])
δMq2 = vec_to_sym(J[:,2])
δMq3 = vec_to_sym(J[:,3])

# So apparently you can do forward diff on NN. Neat
display(δMq1)
println()
display(δMq2)
println()
display(δMq3)
