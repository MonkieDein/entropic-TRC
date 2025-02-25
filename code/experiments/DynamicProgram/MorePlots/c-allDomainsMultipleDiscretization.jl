include("../../../utils.jl")
using CairoMakie
# using LaTeXStrings
using MakieTeX
lQls = 2 .^ [4,8,12] 
lEQl = 10 
parEval = collect(LinRange(0, 1, lEQl*2+1))[2:2:end]

marker = Dict("E"=>:diamond,"VaR"=>:circle,"VaR_over"=>:circle,"nVaR"=>:rect,"dVaR"=>:cross)
col = Dict("E"=>:red,"VaR"=>:blue,"VaR_over"=>:darkred,"nVaR"=>:green,"dVaR"=>:black)
T=100

MultiEvals = load_jld("experiment/run/test/multi_evals_$T.jld2")
f = Figure(resolution=(1200,2400))
for (i,lQl) in enumerate(lQls)
    bound = MultiEvals["$lQl"]["bound"]
    ret = MultiEvals["$lQl"]["ret"]
    for (j,pair) in enumerate(ret) 
        domain, results = pair
        ax = Axis(f[j, i], title="$lQl discretization VaR ($domain)") 
        xlims!(ax,0,1)
        ax.xticks = 0:.2:1
        for (ρ, result) in results
            scatter!(ax,result["α"],result["values"], marker = marker[ρ],markersize = 16,color=(col[ρ], 0.5))
        end
        for (ρ, result) in bound[domain]
            lines!(ax,result["α"],result["values"],color=(col[ρ], 0.5))
        end
    end
end
sideinfo2 = Label(f[8, 2:3], "Quantile level", fontsize = 30)
obj_scatter = [[LineElement(color = (col[ρ], 0.5), lw=4) for ρ in ["VaR","VaR_over"]];
[MarkerElement(color = (col[ρ], 0.5), marker = marker[ρ],markersize = 16) for ρ in ["VaR","VaR_over"]]]
l = Legend(f[8, 1:2],obj_scatter,["q̲ᵈ ","q̄ᵈ ","ρ( π̲  )","ρ( π̄  )"],orientation = :horizontal)

sideinfo = Label(f[:, 0], "Quantile value", rotation = pi/2, fontsize = 30)

save(check_path("fig/mc_test_result/combine-discretize-$T.png"), f)
save(check_path("fig/mc_test_result/combine-discretize-$T.pdf"), f)
# end

