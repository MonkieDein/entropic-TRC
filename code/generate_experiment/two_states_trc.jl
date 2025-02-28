include("../TabMDP.jl")
include("../utils.jl")
using DataFrames
using CSV

# Two states Markov chain
# s ----(1-ϵ);(1) -----> e 
# s ----(ϵ);(1)   -----> s
# e ----(1);(0) -----> e 
ϵ = 0.9

# States Actions initialization
lAl = 1
lSl = 2
A = collect(1:lAl)
S = collect(1:lSl)
s0 = zeros(lSl)
s0[1] = 1.0

# Reward function initialization
R = zeros((lSl,lAl,lSl))
R[1,:,:] .= -0.2

# Transition probability initialization
P = zeros((lSl,lAl,lSl))
P[1,1,1] = ϵ
P[1,1,2] = 1-ϵ
P[2,1,2] = 1


idstatefrom=[]
idaction=[]
idstateto=[]
probability=[]
reward=[]
for s in S
    for a in A
        for s_ in S
            if P[s,a,s_] > 0.0
                push!(idstatefrom,s)
                push!(idaction,a)
                push!(idstateto,s_)
                push!(probability,P[s,a,s_])
                push!(reward,R[s,a,s_])
            end
        end
    end
end
df = DataFrame(idstatefrom = idstatefrom, idaction = idaction, 
idstateto = idstateto, probability = probability, reward = reward)
CSV.write(check_path("experiment/domain/csv/twoS_TRC.csv"), df)
