include("../../../utils.jl")
include("../../../experiment.jl")
using CairoMakie

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
eval_every = Int(n_steps/2)
domains = readdir(mdp_dir)
T = -1
obj = Objective(ρ="VaR", pars=pars[1:end-1], parEval=parEval,T = T)

# set up VI variant
T_VI = ifelse(T==-1,T_inf,T)
obj_VI = Objective(ρ="VaR", pars=pars[1:end-1], parEval=parEval,T = T_VI)

# file to read the value function and policy
lr_settings = fill("",4)
for (k_i,k_exp) in enumerate([4,8,12,1000])
    if k_exp >= 324
        lr_settings[k_i] = "k=0"
    else
        lr_settings[k_i] = "k=1e-$k_exp"
    end
end


# Value Iteartion variant
VI_testfile = "experiment/run/test/evals_$T_VI.jld2"
VI_evals = init_jld(VI_testfile)
VI_ret = simplifyEvals([obj_VI],mdp_dir=mdp_dir,testfile=VI_testfile)
VI_bound = getTargetVaR(init_jld("experiment/run/train/out_$T_VI.jld2"),[obj_VI],mdp_dir=mdp_dir)
col = Dict("Q"=>:red,"DP"=>:blue)

f = Figure(resolution=(1600,2400))  
for (i,lr_setting) in enumerate(lr_settings)  
    Q_out_dir = "experiment/run/train/Q_out/$lr_setting/"
    testfile = "experiment/run/train/Q_evals/$lr_setting-$n_steps.jld2"
    Q_ret = simplifyEvals([obj],mdp_dir=mdp_dir,testfile=testfile)
    Q_bound = getTargetVaR(init_jld(Q_out_dir*"$n_steps.jld2"),[obj],mdp_dir=mdp_dir)
    for (j,pair) in enumerate(VI_ret) 
        domain, results = pair
        ax = Axis(f[j, i], title="κ=$(lr_setting[3:end]) Q-learning ($domain)") 
        for (ρ, result) in results
            scatter!(ax,result["α"],result["values"], marker = :circle,markersize = 16,color=(col["DP"], 0.5))
            scatter!(ax,Q_ret[domain][ρ]["α"],Q_ret[domain][ρ]["values"], marker = :circle,markersize = 16,color=(col["Q"], 0.5))
            xlims!(ax,0,1)
            ax.xticks = 0:.2:1
            max_y = Base.max(maximum(result["values"]),maximum(Q_ret[domain][ρ]["values"]))
            min_y = Base.min(minimum(result["values"]),minimum(Q_ret[domain][ρ]["values"]))
            range_y = max_y - min_y
            ylims!(ax,min_y - 0.05*(range_y),max_y+0.05*abs(range_y))
        end
        for (ρ, result) in VI_bound[domain]
            lines!(ax,result["α"],result["values"],color=(col["DP"], 0.5))
            lines!(ax,Q_bound[domain][ρ]["α"],Q_bound[domain][ρ]["values"],color=(col["Q"], 0.5))
        end
    end
end
sideinfo2 = Label(f[8, 1:4], "Quantile level", fontsize = 30)
sideinfo = Label(f[:, 0], "Quantile value", rotation = pi/2, fontsize = 30)
obj_scatter = [[LineElement(color = (col[val], 0.5), lw=4) for val in ["Q","DP"]];
[MarkerElement(color = (col[val], 0.5), marker = :circle,markersize = 16) for val in ["Q","DP"]]]
l = Legend(f[8, 1:2],obj_scatter,["q̃ᵈ ","q̲ᵈ ","ρ( π̃  )","ρ( π̲  )"],orientation = :horizontal)
save(check_path("fig/mc_test_result/all_algs/Q/all.png"), f)
save(check_path("fig/mc_test_result/all_algs/Q/all.pdf"), f)

