Refactored my notebooks to be less manual. Added additional tests to identify which acquisition functions might perform. Also added a wrapper function to keep the warnings but present them after the processing loops so they were clearer. Note that I've used Claude to help build my testing environment by getting it to suggest my test approaches below. 

# Some surrogate validation 
Built a leave one out validation by refitting the GP on every know value bar one and then predicting that value. Looked at the difference to the mean of that prediction and the median prediction because the mean can skew on outliers. 

# Backtesting
Updated the backtesting a bit so that I can test all the acquisition functions I've defined properly. Using half of the original dataset as a training set and half as a validation set to see how well any change to the acquisition function would predict what I currently know.

# Sweeping hyperparameters
Added a mechanism to hold all bar one dimension and see what gives the largest change in y. Also looking at which dimensions sit right on the `length_scale_bounds` ceiling as those are not going to be providing any useful information. 


**Function 1** — Still on UCB. Switched the modelling target from y to log10|y| after leave-one-out testing showed the surrogate was not useful on raw y (ordering worse than a coin flip), because |y| spans 121 orders of magnitude and eleven of twelve observations were effectively identical to the GP. Tightened the bounds to the observed box, since a working surrogate immediately ran to the pad_fraction corner. 

**Function 2** —  Went from ei/xi=0.01 to ucb/kappa=1.0, briefly to kappa=3.0, then back to 1.0 with x1 held at the incumbent. The kappa=3 step was recorded as a mistake: it was meant to escape proposals pinned on x1's 0.0 floor but still extrapolated on x1 — off the top instead — so the real fix was holding the pinned axis rather than turning the exploration dial.

**Function 3** — Came off max_variance to ucb/kappa=1.0 once every gap on the most useful axis fell below the fitted length-scale (ratio 0.90, down from 1.35 the week it found a hidden peak). Only using X2 in any meaningful way as X0 and X1 are showing up as pinned by length_scale_bounds. 

**Function 4** — Just using posterior mean maximimisation for pure exploitation. Excluded the week-2 −215 observation from the GP fit while keeping it in the record - this point throws everything out and is probably in error. With it out, the same acquisition steps 0.0333 and predicts +0.0828. 

**Function 5** — UCB with kappa==1. Realised that the X domain is the unit cube (doh!) which meant all three collected points were out of bounds and its apparent 53× breakthrough (57,792) wasn't a legitimate result. Excluded them from the fit, since they inflated x3's length-scale 23× and y's std 49×, and reset the search to [0,1]^4. Extended the search from x2 alone to x2 and x3. 

**Function 6** — Updated kappa, moving 0.5 → 0.25 because 0.5 was predicting a loss. 

**Function 7** — Switched to exploit because no kappa predicts a gain — the same criterion that moved function 6 to 0.25 gives 0 here, so it's the value that loses least.

**Function 8** — Kept kappa = 0.25, spent the week on a controlled experiment instead: the three collected points are the three highest y values and all sit at exactly x4 = 0.586936.
