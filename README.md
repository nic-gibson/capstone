# capstone
Black Box Optimisation for Imperial College Professional Certificate in Machine Learning and Artificial Intelligence.


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
* Week 6: The mode switch moved to 2.6→2.7, so KAPPA is 2.5. Added the ACKNOWLEDGE_SIGN_RISK gate.
