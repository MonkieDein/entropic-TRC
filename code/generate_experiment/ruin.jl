include("../TabMDP.jl")
include("../utils.jl")
using DataFrames
using CSV

# Gamblers Ruin MDP
# Winning probability
q = 0.68

# StateIndex = Capital + 1 
# States Actions initialization
lAl = 7 # Bets = [0,1,2,3,4,5,6]
lSl = 8 # Capital = [0,1,2,3,4,5,6,7] #(Terminate when (0 or 7) capital)
A = collect(1:lAl)
S = collect(1:lSl)
s0 = zeros(lSl)
s0[1] = 1.0

# Reward function initialization
R = zeros((lSl,lAl,lSl))
for s in S
    R[s,1,1] = s-1 # Game terminate when not betting; leave with capital
end
R[8,:,:] .= 7 # Game terminate when reach maximum total capital (7)

# Transition probability initialization
P = zeros((lSl,lAl,lSl))
for s in S
    for a in A
        if (a == 1) | (s == 8)  # done with playing or ended at (7) capital
            P[s,a,1] = 1.0
        elseif (a ≤ s)
            P[s,a,Base.min(s+(a-1),lSl)] = q
            P[s,a,Base.max(s-(a-1),1)] = 1-q
        end
    end 
end

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
CSV.write(check_path("experiment/domain/csv/ruin7_TRC.csv"), df)

