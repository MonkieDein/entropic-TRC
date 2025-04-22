# Create MDP domains for HIV problem
include("../utils.jl")
using Distributions
using DataFrames
using CSV
# Action Space: (1 : no treatment, 2 : treatment)
lAl = 2 

# State Space: (Age x CD4 x NumberOfTreatmentsOverLast5Yrs) 
# Age : (1 : 15 ~ 24 , 2 : 25 ~ 34 , 3 : 35 ~ 44 , 4 : 45 ~ 54 , 5 : 55 ~ 64 , 6 : 65 ~ 74 , 7 : 75 ~ 84 , 8 : 85+) 
Ages = [15,25,35,45,55,65,75,85] # Age groups # CD4 groups index = (searchsortedlast.(Ref(Age),list_of_values))

# [Source: 2022 fatality rate from CDC]
# Death Rate by Age group (per 100,000): (1 : 79.5 , 2 : 163.4 , 3 : 255.4 , 4 : 453.3 , 5 : 992.1 , 6 : 1978.7 , 7 : 4708.2 , 8 : 14389.6)
DeathRateByAge = [79.5,163.4,255.4,453.3,992.1,1978.7,4708.2,14389.6]/100000 # Death rate by age group

# [Source: Diana et.al, When Is Immediate Treatment Optimal?]
# CD4 : (1 : <50 , 2 : 50 ~ 99 , 3 : 100 ~ 199 , 4 : 200 ~ 349 , 5 : 350 ~ 499 , 6 : 500 ~ 649 , 7 : 650+)
CD4 = ["<50", "50 ~ 99", "100 ~ 199", "200 ~ 349", "350 ~ 499", "500 ~ 649", "650+"] # CD4 groups
CD4_v = [0, 50, 100, 200, 350, 500, 650] # CD4 groups index = (searchsortedlast.(Ref(CD4_v),list_of_values))
# Death Rate no treatment by CD4 group: (1 : 0.1005 , 2 : 0.02 , 3 : 0.0108 , 4 : 0.006 , 5 : 0.0016 , 6 : 0.001 , 7 : 0.0008)
# Death Rate with treatment by CD4 group: (1 : 0.0167 , 2 : 0.0119 , 3 : 0.0085 , 4 : 0.0039 , 5 : 0.0003 , 6 : 0.0002 , 7 : 0.0002)
DeathRateTreatment = [[0.1005, 0.02, 0.0108, 0.006, 0.0016, 0.001, 0.0008],[0.0167, 0.0119, 0.0085, 0.0039, 0.0003, 0.0002, 0.0002]]  # [1] no treatment, [2] treatment 
# NumberOfTreatmentsOverLastYear : (1 : 0 , 2 : 1 , 3 : 2 , 4 : 3 , 5 : 4 , 6 : 5 , 7 : 6 , 8 : 7, 9 : 8+)
NumberOfTreatmentsOverLast5Yrs = [0,1,2,3,4,5,6,7,8] # Number of treatments over last 5 years # index = (treatment + 1)
TreatmentsGroups = length(NumberOfTreatmentsOverLast5Yrs) # Number of treatment groups
# Average CD4 increase N(μ,25): (1 : -35.25 , 2 : 100 , 3 : 50 , 4 : 40 , 5 : 40 , 6 : 25 , 7 : 20 , 8 : 20, 9 : 0) 
CD4IncreaseAvg = [-35.25,100,50,40,40,25,20,20,0] # CD4 increase average
# Each step indicate 6-months, so 2 steps = 1 year

lSl = length(Ages)*length(CD4)*length(NumberOfTreatmentsOverLast5Yrs)+1 # 
A = collect(1:lAl)
S = collect(1:lSl)
S_map = Dict()
I2S_map = Dict()
for (i,age) in enumerate(Ages)
    for (j,cd) in enumerate(CD4_v)
        for (k,nt) in enumerate(NumberOfTreatmentsOverLast5Yrs)
            index = (i-1)*length(CD4)*length(NumberOfTreatmentsOverLast5Yrs) + (j-1)*length(NumberOfTreatmentsOverLast5Yrs) + k
            S_map[(age,cd,nt)] = index
            I2S_map[index] = (age,cd,nt)
        end
    end
end
s0 = ones(lSl)/(lSl-1)
s0[end] = 0.0 # Initial state distribution, 0.0 for terminal state

# Reward function initialization
R = ones((lSl,lAl,lSl))
R[end,:,:] .= 0.0 # terminal state

# CD4 tranisition
CD4_v_2 = [50, 100, 200, 350, 500, 650, 1500] # upper bound CD4 groups
σ = 25
discretize = 10
cd4_p = zeros((length(NumberOfTreatmentsOverLast5Yrs),length(CD4_v),length(CD4_v_2)))
for (k,incr) in enumerate(CD4IncreaseAvg)
    for (low,high,i) in zip(CD4_v,CD4_v_2,1:length(CD4_v))
        components = [Normal(μ+incr, σ) for μ in collect(range(low, stop=high, length=discretize))]
        weights = fill(1/discretize, discretize)
        mixture = MixtureModel(components, weights)
        cd4_p[k,i,:] = [cdf(mixture, x) for x in CD4_v_2]
    end
end
cd4_p[:,:,length(CD4_v)] .= 1.0
cd4_p[:,:,2:length(CD4_v)] .= (cd4_p[:,:,2:length(CD4_v)]  .- cd4_p[:,:,1:(length(CD4_v)-1)])
# Transition probability initialization
P = zeros((lSl,lAl,lSl))
P[end,:,end] .= 1.0 # terminal state
for (i,age) in enumerate(Ages)
    for (j,cd) in enumerate(CD4_v)
        for (k,nt) in enumerate(NumberOfTreatmentsOverLast5Yrs)
            for a in A
                s = (i-1)*length(CD4)*length(NumberOfTreatmentsOverLast5Yrs) + (j-1)*length(NumberOfTreatmentsOverLast5Yrs) + k
                # println(s,a)
                deathrate = DeathRateTreatment[a][j] + DeathRateByAge[i]
                if i == length(Ages) # too old death rate
                    deathrate += (1 - deathrate)/20
                end
                P[s,a,lSl] = deathrate # terminal state

                survivalrate = 1 - deathrate
                turn_older_probs = (i == length(Ages) ? 0.0 : survivalrate/20) # 10 years
                stay_age_group_probs = survivalrate - turn_older_probs
                
                # number_of_treatment_over_last_5_years 
                if a == 1 # no treatment
                    ks_ = [Base.max(1,k - 1)]
                    tr_prs = [1.0]
                else # with treatment
                    ks_ = [Base.max(2,k),Base.min(k + 1, TreatmentsGroups)]
                    tr_prs = [nt/TreatmentsGroups,1.0-nt/TreatmentsGroups]
                end

                for (k_,tr_pr) in zip(ks_,tr_prs)
                    # for stay age group
                    for (i_,age_pr) in zip( (i:(i+1)),[stay_age_group_probs,turn_older_probs] )
                        if age_pr > 0.0
                            for j_ in 1:length(CD4_v_2)
                                s_ = (i_-1)*length(CD4)*length(NumberOfTreatmentsOverLast5Yrs) + (j_-1)*length(NumberOfTreatmentsOverLast5Yrs) + k_
                                # println(s_,k_,j,j_)
                                P[s,a,s_] += cd4_p[k_,j,j_] * age_pr * tr_pr
                            end
                        end
                    end
                end
            end
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
CSV.write(check_path("experiment/domain/csv/HIV.csv"), df)
