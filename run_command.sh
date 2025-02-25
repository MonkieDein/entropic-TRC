# preliminary
julia requirement.jl

# main code

julia code/others/csv2MDP.jl
julia --threads 8 code/experiments/DynamicProgram/a-multipleDiscretizeExperiment.jl
julia --threads 8 code/experiments/DynamicProgram/b-allAlgorithmsComparison.jl
julia --threads 8 code/experiments/QLearning/a-qLearning.jl
julia --threads 8 code/experiments/QLearning/b-evaluateQlearningPolicies.jl
julia code/experiments/QLearning/MorePlots/b-QValueW1Distance.jl

# optional 
julia code/experiments/DynamicProgram/MorePlots/a-inv2MultipleDiscretization.jl
julia code/experiments/DynamicProgram/MorePlots/b-allAlgosTable.jl
julia code/experiments/DynamicProgram/MorePlots/c-allDomainsMultipleDiscretization.jl
julia code/experiments/DynamicProgram/MorePlots/d-allAlgosScatterPlot.jl
julia code/experiments/QLearning/MorePlots/a-qLearningEvalForinv2.jl
julia code/experiments/QLearning/MorePlots/c-qLearningEvalForAllDomain.jl


