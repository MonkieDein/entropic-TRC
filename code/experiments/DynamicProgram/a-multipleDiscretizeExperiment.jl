# compare finite (100 step) and infinite (-1) horizon VaR objective with different discretization. 
# - VaR uses the lower quantile (j/J) for j in [J-1] 
# - VaR_over uses the upper quantile (j/J) for j in 1:J

include("../../experiment.jl")
using Plots

lQls = 2 .^ [4,6,8,10,12] 
lEQl = 10 
parEval = collect(LinRange(0, 1, lEQl*2+1))[2:2:end]
T_inf = 100
for T in [100]
    multi_file = check_path("experiment/run/test/multi_evals_$T.jld2")
    MultiEvals = Dict()

    mdp_dir = "experiment/domain/MDP/"
    filename = "experiment/run/train/out_$T.jld2"
    testfile = "experiment/run/test/evals_$T.jld2"
    # RUN FINITE HORIZON VALUE ITERATION AND EVALUATION
    for lQl in lQls
        pars = collect(LinRange(0, 1, lQl+1))

        VaRObj = Objective(ρ="VaR", pars=pars[1:end-1], parEval=parEval,T = T) 
        VaR_overObj = Objective(ρ="VaR_over", pars=pars[2:end], parEval=parEval,T = T) 
        objs = [ VaRObj ; VaR_overObj ]  #  

        # Solve Value Iteartions 
        solveVI(objs,mdp_dir = mdp_dir,filename = filename)
        vf = init_jld(filename)
        # Monte Carlo Simulate to evaluate policy performance
        need_eval = true
        if need_eval
            evaluations(vf,objs,ENV_NUM = 10000,T_inf=T_inf,mdp_dir=mdp_dir,testfile=testfile,seed=0)
        end

        # Simplify and Plot the Evaluations
        bound = getTargetVaR(vf,[ VaRObj ; VaR_overObj ],mdp_dir=mdp_dir)
        ret = simplifyEvals(objs,mdp_dir=mdp_dir,testfile=testfile)
        for (domain, results) in ret
            plot(title = ( T==-1 ? "infinite" : "$T")*" horizon policy evaluation $domain", xlabel = "Quantile level", ylabel = "Quantile Value",
            legend=:outerright,xlim=(0,1)) # 
            for (ρ, result) in results
                scatter!(result["α"],result["values"],ms=6, label="π"*ifelse(ρ=="VaR","̲","̄")*" performance",alpha=0.5)
            end
            for (ρ, result) in bound[domain]
                plot!(result["α"],result["values"], label="q"*ifelse(ρ=="VaR","̲","̄")*"ᵈ")
            end
            savefig(check_path("fig/mc_test_result/$domain/$T/discretize/$domain-$T-$lQl.pdf"))
        end
        MultiEvals["$lQl"] = Dict("bound"=>bound, "ret"=>ret)
    end
    save_jld(multi_file,MultiEvals)
end