include("../TabMDP.jl")
include("../utils.jl")
using DataFrames
using CSV

# Two states Markov chain
# s1 ----  (ϵ);(-1)  -----> s1 
# s1 ----(1-ϵ);(1)   -----> s2
# s2 ----  (ϵ);(-0.5)-----> s1 
# s2 ----(1-ϵ);(0)   -----> e
# e  ----  (1);(0)   -----> e

ϵ = 0.5

# States Actions initialization
lAl = 1
lSl = 3
A = collect(1:lAl)
S = collect(1:lSl)
s0 = zeros(lSl)
s0[1] = 1.0

# Reward function initialization
R = zeros((lSl,lAl,lSl))
R[1,1,1] = -1
R[1,1,2] = 1
R[2,1,1] = -0.5

# Transition probability initialization
P = zeros((lSl,lAl,lSl))
P[1,1,1] = ϵ
P[1,1,2] = 1-ϵ
P[2,1,1] = ϵ
P[2,1,3] = 1-ϵ
P[3,1,3] = 1.0

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
CSV.write(check_path("experiment/domain/csv/threeS_TRC.csv"), df)
