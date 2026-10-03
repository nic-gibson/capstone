    Section 1: project overview
        Briefly describe the BBO capstone project and its purpose.
        What is the overall goal of the BBO capstone project? Why is it relevant in real-world ML? What’s the high-level idea?
        How would this BBO capstone project support you in your current or future career?
    Section 2: inputs and outputs

        Clearly state what your model receives and returns.
        What are the inputs (query format, dimensions, constraints, etc.)? What is the expected output (response value, performance signal, etc.)? Include example formats, if possible.
    Section 3: challenge objectives

        Outline what you are trying to achieve within the BBO capstone project.
        Is the goal to minimise or maximise the function(s)? What constraints or limitations must you consider (e.g. number of queries, response delay and unknown function structure)?
    Section 4: technical approach

        Describe the strategies you used across your first three query submissions. You’re encouraged to treat this section as a living record – continue updating it as your approach evolves throughout the BBO capstone project.
        What ML methods or heuristics do you use? Will you model the unknown function? Would you consider using SVMs, regressions or Bayesian techniques? 
        How do you balance exploration and exploitation? What makes your approach thoughtful or unique?



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