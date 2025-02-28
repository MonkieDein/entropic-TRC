include("../../onlineMDP.jl")
include("../../experiment.jl")

# -----------------------------------------------------
# |                 ERM Q-learning                    |
# -----------------------------------------------------
function erm_loss(q_sa::AbstractVector{Float64}, η::Float64, β::AbstractVector{Float64},q_s_::AbstractArray{Float64, 2}, γ::Float64 , R::Float64) 
    target = R .+ (γ .* maximum(q_s_,dims=2))[:,1] # (lQl_, A) -> (lQl_,A -> 1 -> 0) -> (lQl)
    q_sa .-= η .* (exp.(-β .* (target .- q_sa)) .- 1) # (lQl)
end

function exp_erm_loss(q_sa::AbstractVector{Float64}, η::Float64, β::AbstractVector{Float64},q_s_::AbstractArray{Float64, 2}, γ::Float64 , R::Float64) 
    v_s_ = minimum(q_s_,dims=2)[:,1] # (lQl_, A) -> (lQl_,A -> 1 -> 0) -> (lQl)
    q_sa .+= (η .* ((exp.(-β .* R) .* v_s_) .- q_sa) )# (lQl)
end

# Multi threaded
n_threads = Threads.nthreads()
println("running with $n_threads of threads")
n_steps = 100000
eval_every = Int(n_steps/1)

mdp_dir = "experiment/domain/MDP/"
domains = readdir(mdp_dir)
obj = Objective(ρ="ERM", pars=[0.1], parEval=[0.1],T=-1)

# initialize soft kappa parameters we train with
lr_settings = Dict()
lr_settings["erm"] = Dict("loss"=>erm_loss,"v_opt_fun"=>maximum,"pi_opt_fun"=>argmax,"init_q" => 0.0)
lr_settings["exp_erm"] = Dict("loss"=>exp_erm_loss,"v_opt_fun"=>minimum,"pi_opt_fun"=>argmin,"init_q" => 1.0)

seed = 0
for (lr_setting,setting) in lr_settings
    loss_fun = setting["loss"]
    opt_fun = setting["v_opt_fun"]
    pi_fun = setting["pi_opt_fun"]
    q0 = setting["init_q"]

    soln = Dict("$i" => init_jld("experiment/run/train/Q_out/$lr_setting/$i.jld2") for i in eval_every:eval_every:n_steps)
    for d in domains       
        println("$lr_setting-$d")
        domain = d[1:end-5]
        mdp = load_jld(mdp_dir * d)["MDP"]
        
        # Initialize random_seed and learning rate
        Random.seed!(seed)
        lr = Counter(1e-8,expDecay,decay_rate=1e-4,ϵ = 1)

        # Create a vector (S x A) of all possible s,a pair 
        states = repeat(mdp.S, inner=[mdp.lAl]) # (Batch)
        actions = repeat(mdp.A, outer=[mdp.lSl]) # (Batch)
        # Handle invalid actions  
        invalid = [(sum(r) == -Inf) for r in view.(Ref(mdp.R),states,actions,:)]
        q = fill(q0,(mdp.lSl,obj.l,mdp.lAl)) # initialize q value function, invalid action takes -Inf
        for (s,a) in zip(states[invalid],actions[invalid])
            q[s,:, a] .= -Inf
        end
        states = states[.!invalid]
        actions = actions[.!invalid]
        total_valid = length(states)
        # Sample transitions for each valid (s,a) pair
        states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
        rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
        
        # iterate over n update states, for each iteration we update all state action pair once.
        for i in ProgressBar(1:n_steps)
            ind = ((i-1) % eval_every) + 1 # index of the sampled s_ state
            # step and update on loss function
            Threads.@threads for (s,a,s_,r) in collect(zip(states,actions,states_[:,ind],rewards[:,ind]))
                loss_fun((@view q[s,:,a]), lr.ϵ, obj.pars ,(@view q[s_,:,:]),mdp.γ,r)
            end     
            if i % eval_every == 0 # keep value function in a dictionary and resample states_ 
                insert_jld(soln["$i"] , obj.l, domain, obj.ρ,  mdp_out(apply(opt_fun,q,dims=3),apply(pi_fun,q,dims=3),obj.pars))
                states_ = reduce(vcat,(sample_from_transition.(Ref(mdp.P_sample),states, actions,Ref(eval_every)))')  # (Batch, N)
                rewards = getindex.(Ref(mdp.R),states, actions, states_) # (Batch, N)
                GC.gc()
            end
            increase(lr)
        end
        # saved all the value function
        for i in eval_every:eval_every:n_steps  
            save_jld(check_path("experiment/run/train/Q_out/$lr_setting/$i.jld2"),soln["$i"])
        end
    end

end

domain = "gridworld_transient_mdp"
erm_v = init_jld("experiment/run/train/Q_out/erm/$n_steps.jld2")
println("ERM V: ",erm_v["1"][domain]["ERM"]["v"]')
println("ERM Q policy : ",erm_v["1"][domain]["ERM"]["π"]')

exp_erm_v = init_jld("experiment/run/train/Q_out/exp_erm/$n_steps.jld2")
exp_v = exp_erm_v["1"][domain]["ERM"]["v"]'
println("exp ERM V : ",exp_v)
println("-log( exp_V ) / β : ",(-log.(exp_v) ./ obj.pars))
println("exp ERM policy : ",exp_erm_v["1"][domain]["ERM"]["π"]')

vf_T = 10000
vf_obj = Objective( ρ=obj.ρ, T=vf_T, pars=obj.pars)
vf_file = "experiment/run/train/out_$(vf_T).jld2"
vf = solveVI([vf_obj],mdp_dir = mdp_dir,filename = vf_file,cache=false)
println("ERM Value Iteration V : ",vf["$(vf_obj.l)"][domain]["ERM"][1]["v"][1,:])
println("ERM Value Iteration policy : ",vf["$(vf_obj.l)"][domain]["ERM"][1]["π"][1,:])


