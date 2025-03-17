# -----------------------------------------------------
# Infinite Horizon Value Iteration And Q-learning
# -----------------------------------------------------
include("../utils.jl")
include("../experiment.jl")
include("../TabMDP.jl")
using Plots

T=10000
βs = [.1]
obj = Objective( ρ="ERM", T=T, pars=βs, parEval=βs)
mdp_dir = "experiment/domain/MDP/"
filename = "experiment/run/train/out_$(T).jld2"
testfile = "experiment/run/test/evals_$(T).jld2"
domains = readdir(mdp_dir)
# -----------------------------------------------------
# |                 Domain of interest                |
# -----------------------------------------------------
d = "gridworld_transient_mdp.jld2"
domain = d[1:end-5]
mdp = load_jld(mdp_dir * d)["MDP"]
# -----------------------------------------------------
# |                 ERM Value Iteration               |
# -----------------------------------------------------

v = ermVi(mdp,obj);
println("Value Iteration (v):",v[1]["v"][1,:]')
println("Value Iteration (π):",v[1]["π"][1,:]')

# -----------------------------------------------------
# |                 ERM Q-learning                    |
# -----------------------------------------------------
function erm_loss(q_sa::AbstractVector{Float64}, η::Float64, β::AbstractVector{Float64},q_s_::AbstractArray{Float64, 2}, γ::Float64 , R::Float64) 
    target = R .+ (γ .* maximum(q_s_,dims=2))[:,1] # (lQl_, A) -> (lQl_,A -> 1 -> 0) -> (lQl)
    q_sa .-= η .* (exp.(-β .* (target .- q_sa)) .- 1) # (lQl)
end
function exp_erm_loss(q_sa::AbstractVector{Float64}, η::Float64, β::AbstractVector{Float64},q_s_::AbstractArray{Float64, 2}, γ::Float64 , R::Float64) 
    target = exp.(-β .* R) .* minimum(q_s_,dims=2)[:,1] # (lQl_, A) -> (lQl_,A -> 1 -> 0) -> (lQl)
    q_sa .+= (η .* (target .- q_sa) )# (lQl)
end
function flip_exp_erm_loss(q_sa::AbstractVector{Float64}, η::Float64, β::AbstractVector{Float64},q_s_::AbstractArray{Float64, 2}, γ::Float64 , R::Float64) 
    target = exp.(-β .* R) .* maximum(q_s_,dims=2)[:,1] # (lQl_, A) -> (lQl_,A -> 1 -> 0) -> (lQl)
    q_sa .+= (η .* (target .- q_sa) )# (lQl)
end
lr_settings = Dict()
lr_settings["erm"] = Dict("loss"=>erm_loss)
lr_settings["exp_erm"] = Dict("loss"=>exp_erm_loss)
lr_settings["flip_exp_erm"] = Dict("loss"=>flip_exp_erm_loss)
# Multi threaded
n_threads = Threads.nthreads()
println("running with $n_threads of threads")
n_steps = 100000
eval_every = Int(n_steps/10)
seed = 0
Random.seed!(seed)

# initialize Q learning parameter
Q_obj = Objective(ρ="ERM", pars=βs, parEval=βs,T=-1)
# Create a vector (S x A) of all possible s,a pair 
states = repeat(mdp.S, inner=[mdp.lAl]) # (Batch)
actions = repeat(mdp.A, outer=[mdp.lSl]) # (Batch)
# Handle invalid actions  
invalid = [(sum(r) == -Inf) for r in view.(Ref(mdp.R),states,actions,:)]
states = states[.!invalid]
actions = actions[.!invalid]
total_valid = length(states)
# Sample transitions for each valid (s,a) pair
states_,rewards = 0,0
# -----------------------------------------------------
#               Actual Q learning update
# iterate over n update states, 
# for each iteration we update all state action pair once.
# -----------------------------------------------------
lr = Counter(1e-8,expDecay,decay_rate=1e-4,ϵ = 1)
sa_method = "erm"
loss_fun = lr_settings[sa_method]["loss"]
q = zeros(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
for (s,a) in zip(states[invalid],actions[invalid])
    q[s,:, a] .= -Inf
end
for i in ProgressBar(1:n_steps)
    if i % eval_every == 1 # keep value function in a dictionary and resample states_ 
        global states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
        global rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
        GC.gc()
    end
    ind = ((i-1) % eval_every) + 1 # index of the sampled s_ state
    # step and update on loss function
    Threads.@threads for (s,a,s_,r) in collect(zip(states,actions,states_[:,ind],rewards[:,ind]))
        loss_fun((@view q[s,:,a]), lr.ϵ, obj.pars ,(@view q[s_,:,:]),mdp.γ,r)
    end     
    increase(lr)
end
soln = mdp_out(apply(maximum,q,dims=3),apply(argmax,q,dims=3),obj.pars)

println("Q-learning (v):",soln["v"]')
println("Q-learning (π):",soln["π"]')

# -----------------------------------------------------
#               Exp Q learning update
# iterate over n update states, 
# for each iteration we update all state action pair once.
# -----------------------------------------------------
lr = Counter(1e-8,expDecay,decay_rate=1e-4,ϵ = 1)
sa_method = "exp_erm"
loss_fun = lr_settings[sa_method]["loss"]
q = ones(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
for (s,a) in zip(states[invalid],actions[invalid])
    q[s,:, a] .= -Inf
end
for i in ProgressBar(1:n_steps)
    if i % eval_every == 1 # keep value function in a dictionary and resample states_ 
        global states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
        global rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
        GC.gc()
    end
    ind = ((i-1) % eval_every) + 1 # index of the sampled s_ state
    # step and update on loss function
    Threads.@threads for (s,a,s_,r) in collect(zip(states,actions,states_[:,ind],rewards[:,ind]))
        loss_fun((@view q[s,:,a]), lr.ϵ, obj.pars ,(@view q[s_,:,:]),mdp.γ,r)
    end     
    increase(lr)
end
exp_soln = mdp_out(apply(minimum,q,dims=3),apply(argmin,q,dims=3),obj.pars)
println("EXP Q-learning (v):",(-log.(exp_soln["v"]') ./ obj.pars))
println("EXP Q-learning (π):",exp_soln["π"]')

# -----------------------------------------------------
#               Flip Exp Q learning update
# iterate over n update states, 
# for each iteration we update all state action pair once.
# -----------------------------------------------------
lr = Counter(1e-8,expDecay,decay_rate=1e-4,ϵ = 1)
sa_method = "flip_exp_erm"
loss_fun = lr_settings[sa_method]["loss"]
q = -ones(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
for (s,a) in zip(states[invalid],actions[invalid])
    q[s,:, a] .= -Inf
end
for i in ProgressBar(1:n_steps)
    if i % eval_every == 1 # keep value function in a dictionary and resample states_ 
        global states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
        global rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
        GC.gc()
    end
    ind = ((i-1) % eval_every) + 1 # index of the sampled s_ state
    # step and update on loss function
    Threads.@threads for (s,a,s_,r) in collect(zip(states,actions,states_[:,ind],rewards[:,ind]))
        loss_fun((@view q[s,:,a]), lr.ϵ, obj.pars ,(@view q[s_,:,:]),mdp.γ,r)
    end     
    increase(lr)
end
flip_exp_soln = mdp_out(apply(maximum,q,dims=3),apply(argmax,q,dims=3),obj.pars)
println("flip EXP Q-learning (v):",(-log.(-flip_exp_soln["v"]') ./ obj.pars))
println("flip EXP Q-learning (π):",flip_exp_soln["π"]')


