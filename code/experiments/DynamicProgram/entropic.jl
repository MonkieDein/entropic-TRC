# -----------------------------------------------------
# Infinite Horizon Value Iteration And Q-learning
# -----------------------------------------------------
include("../../utils.jl")
include("../../experiment.jl")
include("../../TabMDP.jl")
using Plots

T=10000
βs = [.1]
obj = Objective( ρ="ERM", T=T, pars=βs, δ=1e-10)
meanObj = Objective(ρ="E", pars=[1.0],T = T) # mean

objs = [obj ; meanObj ]  
mdp_dir = "experiment/domain/MDP/"
filename = "experiment/run/train/out_$(T).jld2"
testfile = "experiment/run/test/evals_$(T).jld2"
domains = readdir(mdp_dir)

# -----------------------------------------------------
# |                 ERM Value Iteration               |
# -----------------------------------------------------

domain = domains[1]
mdp = load_jld(mdp_dir * domain)["MDP"]
v = ermVi(mdp,obj);

v[1]["v"][1,:]'
v[1]["π"][1,:]'
# run value iteration for all domains
# vf = solveVI(objs,mdp_dir = mdp_dir,filename = filename,cache=false)

# inspect value function of a domain
# vf["$(obj.l)"][domain][obj.ρ][1]
