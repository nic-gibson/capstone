# Capstone Project - Imperial College Business School Machine Learning & AI

## Overview



This is a black box optimisation project using Bayesian Optimisation.  There are eight functions to be processed with increasing dimensionality  (running from two to 8). The purpose is to optimise a Gaussian Processor in order to emulate the behaviour of the hidden function as closely as possible. 

Bayesian Optimisation is useful in both the machine learning and physical worlds. It is useful when it is expensive to carry out the real operation (a function call or a real world activity such as biological testing). In a SAAS environment this can be useful when identifying optimal configurations for cloud infrastructure/software as measuring the effectiveness of infrastructure would normally require creation and configuration of that infrastructure which is a costly process.

## Inputs & Outputs

At the starting point, we received a certain number of inputs to and outputs from the hidden functions (see below). Every output is a single scalar. Inputs are constrained to be in the range [0, 1]. The initial datasets contained between ten and forty input and output points

| Function No | Degree | Initial Points |
| ----------: | -----: | -------------: |
| 1 | 2 | 10 |
| 2 | 2 | 10 | 
| 3 | 3 | 15 | 
| 4 | 4 | 30 | 
| 5 | 4 | 20 |
| 6 | 5 | 20 |
| 7 | 6 | 30 |
| 8 | 8 | 40 |


## Challenge Objectives

This is a maximisation problem so the objective is to maximise each of the 8 hidden functions. The task is constrained by the lack of information on the nature of the functions. The task starts with a certain number of initial points supplied and, each week, the y value from a proposed Xn values can be requested - once per week. Each function may have entirely different features and some may be much more noisy than others. In the higher order functions (6, 7 and 8) the provided set of initial values provides a very small proportion of the total volume of the problem space. The only known constraint on the Xn values is that they are all between 0.0 and 1.0 


## Technical Approach 

Rather than choosing a single approach, I built multiple acquisition functions and set of tools designed to test those. I built the following acquisition functions

* UCB 
* maximum variance
* pure exploit using just the mean 
* probability of improvement
* expected improvement

I use the initial data to experiment and identify potentially useful acquisition functions. I built tools to use the initial data as training and validation sets and compared each functions performance. I followed a similar approach with the values of hyperparameters (_xi_ and _kappa_). Each week I review the acquisition functions and their hyperparameters to determine if either needs to be changed. At each stage I've used some visulation tools and leave one out evalution to test if I'm still following the right approach to whether exploitation or exploration is required. 




## Week 1 Summary (see notebook for more detail)

* Function 1: ucb with kappa=5.0 and padded bounds. result was y = -3.77e-115.
* Function 2: ucb, kappa=5.0. proposal had x1=1.0505, outside the unit cube.
* Function 3: pi with xi=0.01. proposal returned y = -0.0564, worse than initial best
* Function 4: pi. proposal sat on a bound in every coordinate, including two negative ones.
* Function 5: pi. proposal was outside the cube on x2 and x3.
* Function 6: pi. result of -0.512 was the best so far.
* Function 7: pi. proposal had a negative x4 and returned y = 0.000309, the worst of 31 points.
* Function 8: pi. result was 9.928, 

## Week 2 Summary

Split codebase into multiple modules. Added some visualisation. Worked on
building a testing framework that functioned better. 

* Week 1: ucb with kappa=5.0 and padded bounds. First result was y = -3.77e-115.
* Week 2: Own notebook with KAPPA=3.0 on a linearly rescaled y. Added the backtest, LOO and kappa comparison.
* Week 3: Target changed to log10|y|, floored at -130. LOO R² went from -0.56 to +0.68. Bounds moved to the observed box. Added the hull check.
* Week 4: Added upper_limit=1.0. The mode switch moved from kappa 8→10 to 3→4, and KAPPA=3.0 was kept as the last usable step. Moved to the CSV store.
* Week 5: KAPPA dropped to 2.0 because kappa=3 hit the box corner. The proposal gave y = +0.031, the new incumbent.
* Week 6: The mode switch moved to 2.6→2.7, so KAPPA is 2.5. Added the
  ACKNOWLEDGE_SIGN_RISK gate.

## Week 3 Summary

* Function 1: Target changed to log10|y|, floored at -130. LOO R² went from -0.56 to +0.68. Bounds moved to the observed box. Added the hull check.
* Function 2: Switched to ucb with KAPPA=1.0. Added the paired-regret verdict and a sweep showing a mode switch between kappa 2 and 3.
* Function 3: Refactor only. Added fitting(). Acquisition unchanged.
* Function 4: Same approach. The result was +0.157, the first positive y.
* Function 5: Same approach. The proposal landed exactly on the padded bound at x2 = 1.670022.
* Function 6: Code unchanged. The result of -0.386 was a new best.
* Function 7: Code changes only. Result 1.3271.
* Function 8: No change to approach. The proposal had three coordinates on the 0.0 floor.

## Week 4 Summary

* Function 1: Added upper_limit=1.0. The mode switch moved from kappa 8→10 to 3→4, and KAPPA=3.0 was kept as the last usable step. Moved to the CSV store.
* Function 2: Added upper_limit=1.0. x1's length-scale pinned, so kappa 0 to 2 hit the x1=0 floor. KAPPA=3.0 was tried then reverted. The fix was to hold x1 and scan x0.
* Function 3: Gap-to-length-scale ratio fell from 1.35 to 0.90, so switched to ucb with kappa=1.0. Added upper_limit=1.0.
* Function 4: Added EXCLUDE_FROM_FIT = [0] to drop the -215 point from the fit. Length-scales halved, and the UCB mode switch moved to kappa 1.5 to 2. Moved to the CSV store.
* Function 5: The 6-dp rounding tripped the bounds check, which exposed the [0,1] domain. Bounds rebuilt and the three out-of-domain points excluded from the fit. x0 and x1 held, x2 and x3 searched. Moved to the CSV store.
* Function 6: Added upper_limit=1.0 and the predicted-gain rule, giving KAPPA=0.25. x4 is monotone, so floor contact is expected.
* Function 7: Switched to exploit over the same four axes, because no kappa predicted a gain. Bounds became the true domain. Result 1.4341, a new best.
* Function 8: Bounds became [0,1]. The x4 confound was identified (all three collected points share x4 = 0.586936), so the proposal moves x4 alone to 1.0. Moved to the CSV store.

## Week 5 Summary

Modified my approach to storing weekly data and added the weekly_data directory. All functions now write their proposals to `weekly_data/functionX/input.csv`. On receipt of evaluation this must be saved to output.txt and the `record_results.py` function run. That will write the Y data received to `weekly_data/functionX/output.csv`. Functions now read both the provided data from `initial_data` and the weekly data from `weekly_data`.

* Function 1: KAPPA dropped to 2.0 because kappa=3 hit the box corner. The proposal gave y = +0.031, the new incumbent.
* Function 2: Moved to the CSV store. No change to approach.
* Function 3: KAPPA dropped to 0.5 because 1.0 now performed worse. Added the pairwise scatter plot.
* Function 4: Documentation edits only.
* Function 5: The in-domain result of 2917.48 became the incumbent. A frozen-axis confound was flagged, so the proposal is a test with x0 set to 0.
* Function 6: Switched to max_variance because no kappa above 0 beat the incumbent. The corner proposal [0,0,0,0,0] returned -2.308.
* Function 7: Back to ucb with KAPPA=0.5, as the predicted-gain rule had changed. Result 1.4353.
* Function 8: The x4 test resolved the confound as an artefact. x4 came off the held list and kappa rose to 0.5. The first acquisition-driven proposal in three weeks gave 9.996, a new incumbent.
