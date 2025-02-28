# -----------------------------------------------------
# Infinite Horizon Value Iteration And Q-learning
# -----------------------------------------------------
include("../../utils.jl")
include("../../experiment.jl")
include("../../TabMDP.jl")
using Plots

T=10000
βs = [.1]
obj = Objective( ρ="ERM", T=T, pars=βs)
meanObj = Objective(ρ="E", pars=[1.0],T = T) # mean

objs = [obj ; meanObj ]  
mdp_dir = "experiment/domain/MDP/"
filename = "experiment/run/train/out_$(T).jld2"
testfile = "experiment/run/test/evals_$(T).jld2"
domains = readdir(mdp_dir)

# -----------------------------------------------------
# |                 ERM Value Iteration               |
# -----------------------------------------------------

# run value iteration for all domains
vf = solveVI(objs,mdp_dir = mdp_dir,filename = filename,cache=false)

# inspect value function of a domain
vf["$(obj.l)"][domain][obj.ρ][1]
