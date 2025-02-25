
# Evaluate all the algorithms performance for each parEval. 

include("../../../utils.jl")
include("../../../experiment.jl")
using CairoMakie

lQl = 2^12
lEQl = 10 
pars = collect(LinRange(0, 1, lQl+1))
par_hat= collect(LinRange(0, 1, lQl*2+1)[2:2:end]) # used for QRDQN methods
parEval = collect(LinRange(0, 1, lEQl*2+1))[2:2:end]

marker = Dict("E"=>:diamond,"VaR"=>:circle,"Chow"=>:hexagon,"CVaR"=>:hexagon,"EVaR"=>:star5,"nVaR"=>:rect,"dVaR"=>:star4)
col = Dict("E"=>:red,"VaR"=>:blue,"Chow"=>:brown,"CVaR"=>:magenta,"EVaR"=>:cyan,"nVaR"=>:green,"dVaR"=>:black)
T=100
mdp_dir = "experiment/domain/MDP/"
testfile = "experiment/run/test/evals_$(T).jld2"
# Combine evaluation
meanObj = Objective(ρ="E", pars=[1.0],parEval=parEval,T = T) # mean
VaRObj = Objective(ρ="VaR", pars=pars[1:end-1], parEval=parEval,T = T) # VaR
nVaRObj = Objective(ρ="nVaR", pars=parEval, parEval=parEval,T = T) # nVaR
distVaRObj = Objective(ρ="dVaR", pars=par_hat,parEval=parEval,T = T) # distVaR
EVaRObj = Objective(ρ="EVaR", pars=parEval,parEval=parEval,T = T) # EVaR
ChowObj = Objective(ρ="Chow", pars=pars, parEval=parEval,T = T) # Chow
BaurleCVaRObj = Objective(ρ="CVaR", pars=[1.0], parEval=parEval,δ = 5,T = T) # CVaR (relative significant δ)
objs = [  VaRObj ; nVaRObj;distVaRObj ; meanObj;ChowObj ;BaurleCVaRObj;EVaRObj  ]  #  

risk_name, eval_metric = ("VaR",VaR)
ret = simplifyEvals(objs,mdp_dir=mdp_dir,testfile=testfile,eval_metric = eval_metric)

# generating plots axis 
let ind = 2
    f = Figure(resolution=(800,1200))
    for (domain, results) in ret
        if domain != "inventory2"
            ax = Axis(f[(ind>>1),(ind%2+2)], title="policy evaluation $domain") 
            xlims!(ax,0,1)
            ax.xticks = 0:.2:1
            for obj in objs
                ρ = obj.ρ
                result = results[ρ]
                scatter!(result["α"],result["values"], marker = marker[ρ],markersize=16,color=(col[ρ],0.5))
            end
            ind = ind + 1
        end
    end
    label_x = Label(f[4, 2:3], "Quantile Level", fontsize = 30)
    label_y = Label(f[:, 1], "Quantile Value", rotation = pi/2, fontsize = 30)

    legend_scatter = [MarkerElement(color = (col[obj.ρ], 0.5), marker = marker[obj.ρ],markersize = 16) for obj in objs]
    legend_labels = [ifelse(obj.ρ=="VaR","Alg 1",ifelse(obj.ρ=="dVaR","VaR-IQN",obj.ρ)) for obj in objs]

    l = Legend(f[1, 1],legend_scatter,legend_labels,orientation = :vertical)

    save(check_path("fig/mc_test_result/all_algs/$risk_name/no-INV2-combine-VI.pdf"), f)
end

