include("../../onlineMDP.jl")


function quant_v(q_s_::AbstractArray{Float64, 2}, γ::Float64 , R::Float64)
    return R .+ (γ .* maximum(q_s_',dims=1)) # (lQl_, A) -> (A,lQl_) ->(A -> 1, lQl_)
end

function qr_loss(q_sa::AbstractVector{Float64}, ϵ::Float64, pars::AbstractVector{Float64},target::Matrix{Float64};k=0.01) #k is not used
    q_sa .+= (ϵ .* (pars .- mean(target .< q_sa,dims=2)[:,1])) # (lQl, lQl_ -> 0)
end

function soft_qr_loss(q_sa::AbstractVector{Float64}, ϵ::Float64, pars::AbstractVector{Float64},target::Matrix{Float64};k=0.01)
    x = (target .- q_sa)
    factors = ifelse.( (abs.(x) .<= k) , (abs.(x) ./ k) , ((1 - k^2) .+ (abs.(x) .* k) ))
    q_sa .+= (ϵ .* apply(mean, (pars .- (x .< (0))) .* factors ,dims=2)) # (lQl, lQl_ -> 0)
end

function destandardize_q_output(q_ori,pars::Vector{Float64};reward_range::Float64=1,biasR::Float64=0)
    q = sort(q_ori, dims=2) # 
    destandardize_v = (apply(maximum,q,dims=3) * reward_range) .+ biasR
    return mdp_out(destandardize_v,apply(argmax,q,dims=3),pars)
end

# Initialize discretization
lQl = 2^8
lEQl = 10 
pars = collect(LinRange(0, 1, lQl+1))
parEval = collect(LinRange(0, 1, lEQl*2+1))[2:2:end]

# Initialize Q learning informations
T = -1
n_steps = 20000
eval_every = Int(n_steps/100)
mdp_dir = "experiment/domain/MDP/"
domains = readdir(mdp_dir)
obj = Objective(ρ="VaR", pars=pars[1:end-1], parEval=parEval,T=T)
seed = 0

n_threads = Threads.nthreads()
println("running with $n_threads of threads")

# initialize soft kappa parameters we train with
lr_settings = Dict()
for k_exp in [4,8,12,1000] 
    if k_exp >= 324
        lr_settings["k=0"] = Dict("loss"=>qr_loss,"k"=>10.0^(-k_exp))
    else
        lr_settings["k=1e-$k_exp"] = Dict("loss"=>soft_qr_loss,"k"=>10.0^(-k_exp))
    end
end

for (lr_setting,setting) in lr_settings
    loss_fun = setting["loss"]
    k = setting["k"]    
    soln = Dict("$i" => init_jld("experiment/run/train/Q_out/$lr_setting/$i.jld2") for i in eval_every:eval_every:n_steps)
    for d in domains       
        println("$lr_setting-$d")
        domain = d[1:end-5]
        mdp = load_jld(mdp_dir * d)["MDP"]
        
        # Initialize random_seed and learning rate
        Random.seed!(seed)
        lr = Counter(1e-4,expDecay,decay_rate=3e-4,ϵ = 1e2)

        # standardize domain reward function
        all_rewards = filter(x -> x != Inf && x != -Inf, mdp.R[mdp.P .> 0] )
        reward_range = maximum(all_rewards)-minimum(all_rewards)
        biasR =  minimum(all_rewards)/(1-mdp.γ)
        mdp.R .-= minimum(all_rewards)
        mdp.R ./= reward_range

        # Create a vector (S x A) of all possible s,a pair 
        states = repeat(mdp.S, inner=[mdp.lAl]) # (Batch)
        actions = repeat(mdp.A, outer=[mdp.lSl]) # (Batch)
        # Handle invalid actions  
        invalid = [(sum(r) == -Inf) for r in view.(Ref(mdp.R),states,actions,:)]
        q = zeros(mdp.lSl,obj.l,mdp.lAl) # initialize q value function, invalid action takes -Inf
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
                loss_fun((@view q[s,2:end,a]), lr.ϵ, (@view obj.pars[2:end]) ,quant_v((@view q[s_,:,:]),mdp.γ,r),k=k)
            end     
            if i % eval_every == 0 # keep value function in a dictionary and resample states_ 
                insert_jld(soln["$i"] , obj.l, domain, obj.ρ,  destandardize_q_output(q,obj.pars;reward_range=reward_range,biasR=biasR))
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
