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
f = Figure(resolution=(1200,400))
domain = "inventory2"

for (i,lQl) in enumerate(lQls)
    bound = MultiEvals["$lQl"]["bound"][domain]
    results = MultiEvals["$lQl"]["ret"][domain]
    ax = Axis(f[1, i], title="J=$lQl discretization VaR",titlefont="CMU Serif", titlesize=24) 
    xlims!(ax,0,1)
    ax.xticks = 0:.2:1
    for (ρ, result) in results
        scatter!(ax,result["α"],result["values"], marker = marker[ρ],markersize = 16,color=(col[ρ], 0.5))
    end
    for (ρ, result) in bound
        lines!(ax,result["α"],result["values"],color=(col[ρ], 0.5))
    end
    if i>1
        hideydecorations!(ax, grid = false)
    end
end
sideinfo2 = Label(f[2, 2:3], "Quantile level", fontsize = 22,font="CMU Serif")
obj_scatter = [[LineElement(color = (col[ρ], 0.5), lw=4) for ρ in ["VaR","VaR_over"]];
[MarkerElement(color = (col[ρ], 0.5), marker = marker[ρ],markersize = 16) for ρ in ["VaR","VaR_over"]]]
l = Legend(f[2, 1:2],obj_scatter,["q̲ᵈ ","q̄ᵈ ","ρ( π̲  )","ρ( π̄  )"],orientation = :horizontal)

sideinfo = Label(f[1, 0], "      Quantile value      ", rotation = pi/2, fontsize = 22,font="CMU Serif")


save(check_path("fig/mc_test_result/$domain/$T/$domain-combine-discretize.png"), f)
save(check_path("fig/mc_test_result/$domain/$T/$domain-combine-discretize.pdf"), f)

