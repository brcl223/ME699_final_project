#Attribution: Hasan Poonawala
import Pkg;
Pkg.activate(@__DIR__);
using LinearAlgebra, Plots, Random
# object that is an LTI system
struct LTIsys
  A::Array{Float64,2}
  B::Array{Float64,2}
  C::Array{Float64,2}
  Rw::Array{Float64,2}
  Rv::Array{Float64,2}
end
include("filter_update.jl")
