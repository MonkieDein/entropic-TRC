include("../../../utils.jl")
include("../../../experiment.jl")
using Plots

# set up discretization for quantile and evaluations
lQl = 2^8
lEQl = 10 
pars = collect(LinRange(0, 1, lQl+1))
par_hat= collect(LinRange(0, 1, lQl*2+1)[2:2:end]) # used for QRDQN methods
parEval = collect(LinRange(0, 1, lEQl*2+1))[2:2:end]

# set up discretization for mdp Q-learning evaluations
ENV_NUM = 10000
seed=0
T_inf = 100

# set up discretization for mdp Q-learning evaluations
mdp_dir = "experiment/domain/MDP/"
n_steps = 20000
eval_every = Int(n_steps/100)
domains = readdir(mdp_dir)
T = -1
obj = Objective(ρ="VaR", pars=pars[1:end-1], parEval=parEval,T = T)

# set up DP variant
T_VI = ifelse(T==-1,T_inf,T)
obj_VI = Objective(ρ="VaR", pars=pars[1:end-1], parEval=parEval,T = T_VI)

# initialize soft kappa parameters we train with
lr_settings = Dict()
for k_exp in [4,8,12,1000] 
    if k_exp >= 324
        lr_settings["k=0"] = Dict("k"=>10.0^(-k_exp))
    else
        lr_settings["k=1e-$k_exp"] = Dict("k"=>10.0^(-k_exp))
    end
end


W1_distance = Dict()
for d in domains
    domain = d[1:end-5]
    VI_q = getTargetVaR(init_jld("experiment/run/train/out_$T_VI.jld2"),[obj_VI],mdp_dir=mdp_dir)
    W1_distance[domain] = Dict()
    plot(title = "Q-learning vs DP ($domain)",ylabel="Wasserstein-1 Distance",xlabel ="Number of Samples",
    titlefontsize = 18,guidefontsize = 16,legendfontsize = 14,tickfontsize=10)
    for (lr_setting,setting) in lr_settings
        println("$lr_setting-$domain")
        Q_out_dir = "experiment/run/train/Q_out/$lr_setting/"
        W1_distance[domain][lr_setting] = Dict()
        W1_distance[domain][lr_setting]["step"] = collect(eval_every:eval_every:n_steps )
        dist = []
        xlims!(0,n_steps*1.1)
        for i in W1_distance[domain][lr_setting]["step"]
            Q_q = getTargetVaR(init_jld(Q_out_dir*"$i.jld2"),[obj],mdp_dir=mdp_dir)
            push!(dist, mean(abs.(Q_q[domain]["VaR"]["values"] .- VI_q[domain]["VaR"]["values"])))
        end
        W1_distance[domain][lr_setting]["distance"] = dist
        plot!(W1_distance[domain][lr_setting]["step"], dist, label="κ = $(lr_setting[3:end])", lw=2)
    end
    savefig(check_path("fig/mc_test_result/all_algs/Q_learning_error/$domain.pdf"))
end


