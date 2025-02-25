# Q-learning for Quantile MDPs: A Decomposition, Performance, and Convergence Analysis

The code can also be found in ```https://anonymous.4open.science/r/VaR-Q-learning```

## Preliminary

(1) Install Julia and VSCode.

(2) Open VSCode and then open **[This Folder]**, open Terminal in VSCode (Menubar-Terminal >> New Terminal (Ctrl + Shift + `)) if does not already open.

(3) ```julia requirement.jl``` :  Install all required julia libraries.


## Run the experiment code

**POSSIBLE ERROR**: `textsize` has been renamed to `fontsize` in Makie v0.19. Please change all occurrences of `textsize` to `fontsize` or revert back to an earlier version. Depends on julia library `Makie` version please rename the argument for all optional codes under MorePlots.

**MANUAL MODIFICATION**: Most main code run with **8** threads, which may speed up the process for certain algorithms. Could replace **8** to any appropriate number of threads that the users computer has. *Q-learning result may vary slightly due to random sampling and multi-threading*.

### Option A : Run all the code with ```run_command.sh``` 

### Option B : Order to run the code (Alternatively)
(1) ```julia code/others/csv2MDP.jl``` : Turn MDP domains CSV into MDP objects. (Create ./experiment/domain/MDP/)

(2) ```julia --threads 8 code/experiments/DynamicProgram/a-multipleDiscretizeExperiment.jl``` :
Run VaR DP for multiple discretizations. (Create ./fig/mc_test_result/[domain_name]/)

(3) ```julia --threads 8 code/experiments/DynamicProgram/b-allAlgorithmsComparison.jl``` :
Run all other algorithms and compare their performance with VaR-DP. (Create ./fig/mc_test_result/all_algs/VaR/)

(4) ```julia --threads 8 code/experiments/QLearning/a-qLearning.jl``` :
Train soft quantile qLearning for different parameter of k. (Create ./experiment/run/train/Q_out/)

(5) ```julia --threads 8 code/experiments/QLearning/b-evaluateQlearningPolicies.jl``` :
Evaluate qLearning and compare its value function and policy performance against DP variants. (Create ./fig/mc_test_result/all_algs/Q/)

(6) ```julia code/experiments/QLearning/MorePlots/b-QValueW1Distance.jl``` :
Compute the W1-Distance between qLearning and DP value function. (Create ./fig/mc_test_result/all_algs/Q_learning_error/)

Other optionals (merging plots and table)
- ```julia code/experiments/DynamicProgram/MorePlots/a-inv2MultipleDiscretization.jl``` : {16,256,4096} Discretize performance of $\bar{q}$ and $\underline{q}$ for inventory2 (Create ./fig/mc_test_result/inventory2/inventory2-combine-discretize.*).
- ```julia code/experiments/DynamicProgram/MorePlots/b-allAlgosTable.jl``` : Generate table that evaluate 25% performance for all algorithms across all domains.
- ```julia code/experiments/DynamicProgram/MorePlots/c-allDomainsMultipleDiscretization.jl``` : {16,256,4096} Discretize performance of $\bar{q}$ and $\underline{q}$ for all domains (Create ./fig/mc_test_result/combine-discretize-100.*).
- ```julia code/experiments/DynamicProgram/MorePlots/d-allAlgosScatterPlot.jl``` : {16,256,4096} Discretize performance of $\bar{q}$ and $\underline{q}$ for all domains except for inventory2 (Create ./fig/mc_test_result/all_algs/VaR/4096/no-INV2-combine-VI.*).
- ```julia code/experiments/QLearning/MorePlots/a-qLearningEvalForinv2.jl``` : k={1e-4,1e-8,1e-12,0} $\tilde{q}$ and $\underline{q}$ for inventory2 (Create ./fig/mc_test_result/all_algs/Q/inventory2.*).
- ```julia code/experiments/QLearning/MorePlots/c-qLearningEvalForAllDomain.jl``` : k={1e-4,1e-8,1e-12,0} $\tilde{q}$ and $\underline{q}$ for all domains (Create ./fig/mc_test_result/all_algs/Q/all.*).


## File Structure
- **code/**
    - *experiment.jl* : General functions for experiments, includes solveVI, evaluations, simplifyEvals and getTargetVaR.
    - *onlineMDP.jl* : Functions to execute policy in a monte carlo simulation. Type of policies include (Markov, QuantileDependent) as well as their time dependent variant.
    - *riskMeasure.jl* : Functions to create a discrete random variable and compute their risk (mean, min, max, q⁻, VaR, CVaR, ERM, EVaR).
    - *TabMDP.jl* : Define MDP and Objective structure. Solve nested, quantile, ERM, EVaR, and distributional (markovQuantile) Dynamic Program (DP) a.k.a Value Iteration (VI).
    - *utils.jl* : Commonly used functions for checking directory, decaying coefficient function and multi-dimensions function-applicator.
    - **others/**
        - *csv2MDP.jl* : Code to convert csv MDP files to MDP objects.
    - **experiments/**
        - **DynamicProgram/**
            - *a-multipleDiscretizeExperiment.jl* : For multiple quantile discretization {16,64,256,1024,4096}, run VaR-DP-underApprox and VaR-DP-overApprox, then evaluate their respective policies.
            - *b-allAlgorithmsComparison.jl* : Solve the optimal policy for each of the algorithm with their respective VI, then evaluate their performance.
            - **MorePlots/** : Code to combine multiple plots as subplots to generate a more compact plot.
        - **QLearning/**
            - *a-qLearning.jl* : Run soft quantile Q learning with different kappa parameters {1e-4,1e-8,1e-12,0}.
            - *b-evaluateQlearningPolicies.jl* : Evaluate Q-learning policy, and compare it with the DP variant.
            - **MorePlots/** : Code to combine multiple plots as subplots to generate a more compact plot.
- **experiment/**
    - **domain/**
        - **csv/** : CSV file containing domain transition and reward.
        - *domains_info.csv* : CSV file containing discount factor and initial state.
        - **MDP/** : MDP JLD2 files (Generate from: ```julia code/others/csv2MDP.jl```)
- **fig/mc_test_result/**  
    - **all_algs/**
        - **VaR/** : Scatter plot of all algorithms performance in each domain. (Generate from: ```julia code/experiments/DynamicProgram/b-allAlgorithmsComparison.jl```)
        - **Q/** : Compare q learning $\tilde{q}$ and dp $\underline{q}$ value function and their $\tilde{\pi}$ and $\underline{\pi}$ policies performances, given kappa parameter.(Generate from: ```julia code/experiments/QLearning/b-evaluateQlearningPolicies.jl```)
        - **Q_learning_error/** : Wasserstein-1 Distance plot of Q learning $\tilde{q}$ and dp $\underline{q}$ value function.(Generate from: ```julia code/experiments/QLearning/MorePlots/b-QValueW1Distance.jl```)
    - **[domain_name]/** : Compare over-approximate dp $\bar{q}$ and under-approximate dp $\underline{q}$ value function and their $\bar{\pi}$ and $\underline{\pi}$ policies performances. (Generate from: ```julia code/experiments/DynamicProgram/a-multipleDiscretizeExperiment.jl```)
