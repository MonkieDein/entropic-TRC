include("../../onlineMDP.jl")
include("../../riskMeasure.jl")

# -----------------------------------------------------
# |                 ERM Q-learning                    |
# -----------------------------------------------------
function erm_loss(q_sa::AbstractVector{Float64}, η::Float64, β::AbstractVector{Float64},target) 
    q_sa .-= η .* (exp.(-β .* (target .- q_sa)) .- 1) # (lQl)
end

function exp_erm_loss(q_sa::AbstractVector{Float64}, η::Float64, β::AbstractVector{Float64},target) 
    q_sa .+= (η .* (exp.(-β .* target) .- q_sa) ) # (lQl)
end

beta = [0.5]
samples = vcat(collect(-10:-1),collect(1:10))
X = distribution(samples)
println("True ERM: $(ERMs(X,beta)[1])")

lr = Counter(1e-4,expDecay,decay_rate=3e-4,ϵ = 1)
erm_pred = [0.0]
for i in 1:100000
    for j in samples
        erm_loss(erm_pred,lr.ϵ,beta,j)
        increase(lr)
    end
end
println("ERM stochastic approximation : $(erm_pred[1])")

lr = Counter(1e-4,expDecay,decay_rate=3e-4,ϵ = 1)
exp_erm_pred = [0.0]
for i in 1:10000
    for j in samples
        exp_erm_loss(exp_erm_pred,lr.ϵ,beta,j)
        increase(lr)
    end
end
println("ERM stochastic approximation via exp: $(-log(exp_erm_pred[1])/beta[1])")



