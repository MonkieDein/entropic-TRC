include("../../utils.jl")
include("../../experiment.jl")
using Plots

# set up discretization for quantile and evaluations
lQl = 2^8
lEQl = 10 
pars = collect(LinRange(0, 1, lQl+1))
parEval = collect(LinRange(0, 1, lEQl*2+1))[2:2:end]

# set up parameter for mdp Q-learning evaluations
ENV_NUM = 10000
seed=0
T_inf = 100
T = -1
n_steps = 20000
mdp_dir = "experiment/domain/MDP/"
domains = readdir(mdp_dir)
obj = Objective(ρ="VaR", pars=pars[1:end-1], parEval=parEval,T = T)
# set up objective for DP variant
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

for (lr_setting,setting) in lr_settings
    Q_out_dir = "experiment/run/train/Q_out/$lr_setting/"
    # run policy evaluation
    filename = Q_out_dir * "$(n_steps).jld2"
    vf = init_jld(filename)
    i = n_steps
    testfile = "experiment/run/train/Q_evals/$lr_setting-$i.jld2"
    evaluations(vf,[obj],ENV_NUM = ENV_NUM,mdp_dir=mdp_dir,testfile=testfile,seed=0,quant_ϵ=1e-14)
    evals = init_jld(testfile)
    # Simplify Evaluation and obtain Q learning Q_value function
    Q_ret = simplifyEvals([obj],mdp_dir=mdp_dir,testfile=testfile)
    Q_bound = getTargetVaR(init_jld(Q_out_dir*"$i.jld2"),[obj],mdp_dir=mdp_dir)

    # Value Iteartion variant (experiments/DynamicProgram/a-multipleDiscretizeExperiment.jl must be run prior to this )
    VI_testfile = "experiment/run/test/evals_$T_VI.jld2"
    VI_evals = init_jld(VI_testfile)
    VI_ret = simplifyEvals([obj_VI],mdp_dir=mdp_dir,testfile=VI_testfile)
    VI_bound = getTargetVaR(init_jld("experiment/run/train/out_$T_VI.jld2"),[obj_VI],mdp_dir=mdp_dir)

    # generate plot for each domain
    for (domain, results) in VI_ret
        plot(title = "κ=$(lr_setting[3:end]) Q-learning ($domain)",dpi=1200, xlabel = "Quantile level", ylabel = "Quantile Value",legend=:outerright) # 
        for (ρ, result) in results
            scatter!(result["α"],result["values"], m = :circle,ms=6, label="ρ( π̲  )",alpha=0.5)
            scatter!(Q_ret[domain][ρ]["α"],Q_ret[domain][ρ]["values"], m = :star4,ms=6, label="ρ( π̃  )",alpha=0.5)
            xlims!(0,1)
            max_y = Base.max(maximum(result["values"]),maximum(Q_ret[domain][ρ]["values"]))
            min_y = Base.min(minimum(result["values"]),minimum(Q_ret[domain][ρ]["values"]))
            range_y = max_y - min_y
            ylims!(min_y - 0.05*(range_y),max_y+0.05*abs(range_y))
        end
        for (ρ, result) in VI_bound[domain]
            lim_val = (parEval[1] .<= result["α"]) .& (result["α"] .<= parEval[end])
            plot!(result["α"],result["values"], label="q̲ᵈ")
            plot!(Q_bound[domain][ρ]["α"],Q_bound[domain][ρ]["values"], label="q̃ᵈ")
        end
        savefig(check_path("fig/mc_test_result/all_algs/Q/$lr_setting-$domain.png"))
    end

end
