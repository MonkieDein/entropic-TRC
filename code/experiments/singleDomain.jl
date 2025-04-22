# -----------------------------------------------------
# Infinite Horizon Value Iteration And Q-learning
# -----------------------------------------------------
include("../utils.jl")
include("../experiment.jl")
include("../TabMDP.jl")
using Plots

T=10000
α = 0.2 # in (0,1)
βs = [2.0,3.3]
obj = Objective( ρ="ERM", T=T, pars=βs, parEval=βs)
mdp_dir = "experiment/domain/MDP/"
filename = "experiment/run/train/out_$(T).jld2"
testfile = "experiment/run/test/evals_$(T).jld2"
domains = readdir(mdp_dir)
# -----------------------------------------------------
# |                 Domain of interest                |
# -----------------------------------------------------
d = "threeS_TRC.jld2"
domain = d[1:end-5]
mdp = load_jld(mdp_dir * d)["MDP"]
# -----------------------------------------------------
# |                 function compute h values         |
# -----------------------------------------------------
function compute_h(v,s0,α) # v_i["α"] is actually the exponential risk β
    if v isa AbstractDict # for q learning output (Dictionary)
        v["ERM_value"] = []
        v["h_value"] = []
        for (bi,β) in enumerate(v["α"])
            erm_val = ERMs(distribution(soln["v"][:,bi],s0),[β])[1]
            h_value = (erm_val + (log(α)/β))
            push!(v["ERM_value"],erm_val)
            push!(v["h_value"],h_value)
        end       
    else  # for value iteration output (Vector)
        for v_i in v
            erm_val = ERMs(distribution(v_i["v"][1,:],s0),[v_i["α"]])[1]
            h_value = (erm_val + (log(α)/v_i["α"]))
            v_i["ERM_value"] = erm_val
            v_i["h_value"] = h_value
        end
    end
end
# uniform distribution over all non-sink states
u_s0 = ones(mdp.lSl) ./ (mdp.lSl-1)
u_s0[end] = 0 # Sink state
output_values = Dict()
# -----------------------------------------------------
# |                 ERM Value Iteration               |
# -----------------------------------------------------

v = ermVi(mdp,obj);
compute_h(v,u_s0,α);
output_values["VI"] = v

println("Value Iteration (v):",[v["v"][1,:] for v in output_values["VI"]])
println("Value Iteration (π):",[v["π"][1,:] for v in output_values["VI"]])
println("Value Iteration (ERM):",[v["ERM_value"] for v in output_values["VI"]])
println("Value Iteration (h):",[v["h_value"] for v in output_values["VI"]])
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
invalid_states = states[invalid]
invalid_actions = actions[invalid]
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
lr =  Counter(1e-8,harmonicDecay,decay_rate= 1e-2,ϵ = .1)# Counter(1e-8,expDecay,decay_rate=1e-4,ϵ = 1) #
sa_method = "erm"
loss_fun = lr_settings[sa_method]["loss"]
q = zeros(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
for (s,a) in zip(invalid_states,invalid_actions)
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
compute_h(soln,u_s0,α);
output_values["ERM Q"] = soln

for (keys,values) in output_values["ERM Q"]
    println("ERM Q-learning $(keys):",values)
end

# # -----------------------------------------------------
# #               Exp Q learning update
# # iterate over n update states, 
# # for each iteration we update all state action pair once.
# # -----------------------------------------------------
# lr = Counter(1e-8,harmonicDecay,decay_rate= 1e-2,ϵ = .1)# Counter(1e-8,expDecay,decay_rate=1e-4,ϵ = 1)
# sa_method = "exp_erm"
# loss_fun = lr_settings[sa_method]["loss"]
# q = ones(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
# for (s,a) in zip(invalid_states,invalid_actions)
#     q[s,:, a] .= Inf
# end
# for i in ProgressBar(1:n_steps)
#     if i % eval_every == 1 # keep value function in a dictionary and resample states_ 
#         global states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
#         global rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
#         GC.gc()
#     end
#     ind = ((i-1) % eval_every) + 1 # index of the sampled s_ state
#     # step and update on loss function
#     Threads.@threads for (s,a,s_,r) in collect(zip(states,actions,states_[:,ind],rewards[:,ind]))
#         loss_fun((@view q[s,:,a]), lr.ϵ, obj.pars ,(@view q[s_,:,:]),mdp.γ,r)
#     end     
#     increase(lr)
# end
# exp_soln = mdp_out(apply(minimum,q,dims=3),apply(argmin,q,dims=3),obj.pars)

# println("EXP Q-learning (v):",(-log.(exp_soln["v"]') ./ obj.pars))
# println("EXP Q-learning (π):",exp_soln["π"]')

# # -----------------------------------------------------
# #               Flip Exp Q learning update
# # iterate over n update states, 
# # for each iteration we update all state action pair once.
# # -----------------------------------------------------
# lr = Counter(1e-8,harmonicDecay,decay_rate= 1e-2,ϵ = 0.1) # Counter(1e-8,expDecay,decay_rate=1e-4,ϵ = 1)
# sa_method = "flip_exp_erm"
# loss_fun = lr_settings[sa_method]["loss"]
# q = -ones(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
# for (s,a) in zip(invalid_states,invalid_actions)
#     q[s,:, a] .= -Inf
# end
# for i in ProgressBar(1:n_steps)
#     if i % eval_every == 1 # keep value function in a dictionary and resample states_ 
#         global states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
#         global rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
#         GC.gc()
#     end
#     ind = ((i-1) % eval_every) + 1 # index of the sampled s_ state
#     # step and update on loss function
#     Threads.@threads for (s,a,s_,r) in collect(zip(states,actions,states_[:,ind],rewards[:,ind]))
#         loss_fun((@view q[s,:,a]), lr.ϵ, obj.pars ,(@view q[s_,:,:]),mdp.γ,r)
#     end     
#     increase(lr)
# end
# flip_exp_soln = mdp_out(apply(maximum,q,dims=3),apply(argmax,q,dims=3),obj.pars)
# println("flip EXP Q-learning (v):",(-log.(-flip_exp_soln["v"]') ./ obj.pars))
# println("flip EXP Q-learning (π):",flip_exp_soln["π"]')


