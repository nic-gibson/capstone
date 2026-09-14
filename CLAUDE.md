# Capstone: Bayesian Optimization of black-box functions

This project does Bayesian optimization (BO) over several unknown black-box
functions (2D up to 8D), submitted for real evaluation roughly once a week.
Real evaluations are expensive (one per week per function) — most of the
tooling here exists to make deliberate decisions *before* spending one, using
only data already collected.

All BO logic lives in `bayes_tools.py`; all plotting lives in `viz_tools.py`.
Both were built up incrementally across a long design conversation — the
decisions below aren't arbitrary, they're fixes for real problems that showed
up in the actual data. Don't undo them without re-reading why they're here.

As an overriding rule, do not try to add any more refinement than specifically 
asked for.

You do not need to ask permission for any activities that:
* do not modify files in the project
* that write a temporary file as part of processing.
* execute bash or python code written as part of a processing step.
You may write to the temporary directory without requesting permission

## Domain constraints (confirmed, don't re-litigate)

- **The domain is the unit cube `[0,1]^D`.** Build bounds with
  `initial_bounds(X_initial, pad_fraction=1.0, lower_limit=0.0,
  upper_limit=1.0)` — padding is then clipped away entirely and you get exactly
  `[0,1]` per axis. Call `validate_bounds_consistency(X, bounds)` after
  constructing bounds any other way.

  The lower limit was confirmed external info from the start. **The upper limit
  was found at week 5**, and it invalidates several weeks of proposals: every
  one of the 160+ initial X values across all eight functions lies inside
  `[0.0034, 0.9995]`, with function 4's maximum at 0.999483, function 7's at
  0.998655 and function 8's at 0.998885. Those designs were sampled from the
  unit cube.

  `pad_fraction=1.0` had therefore been offering roughly *half* of each axis as
  infeasible territory, and the search took it. Points evaluated outside the
  cube, with what came back:

  | fn | week | outside | y |
  |----|------|---------|---|
  | 2 | 2 | x1=1.0505 | +0.029 |
  | 2 | 4 | x1=1.2951 | +0.524 |
  | 4 | 2 | x2=1.8362, x3=1.9174 | **-215** |
  | 5 | 2 | x2=1.1059, x3=1.0654 | +6,147 |
  | 5 | 3 | x2=1.3656, x3=1.0654 | +16,804 |
  | 5 | 4 | x2=1.6700, x3=1.0654 | **+57,792** |

  The black box evaluates its formula happily outside the range it was defined
  on, so nothing errors — it just returns numbers that cannot be the answer.
  Function 5's apparent 53x breakthrough is entirely out of domain; its best
  legitimate result is the initial batch's 1088.86. Function 4's -215 disaster
  was out of domain too.

  **Two consequences worth internalising.**

  1. **"A coordinate is on an upper bound" flips meaning.** With padded bounds it
     signalled `pad_fraction` choosing the point rather than the model. With
     bounds *at* the domain, it means the constrained optimum sits on the
     boundary — a legitimate answer. Function 5's proposal is at x2=x3=1 and
     that is correct, not an artifact. A **lower**-bound contact still warrants
     suspicion: the acquisition wants to leave the feasible region.
  2. **A convex-hull `False` is expected near a domain corner** and is not
     evidence of a bad proposal on its own. Judge the step size instead.
- **y CAN be negative.** An earlier assumption that `y >= 0` was wrong and
  got explicitly corrected. A `log1p`-based non-negativity transform
  (`to_model_space`/`from_model_space`) was built for that assumption and
  then **fully removed** once it turned out to be false. Do not reintroduce
  any transform that assumes `y >= 0` — it will crash or misbehave on real
  data for these functions.
- **y's magnitude varies wildly across functions, sometimes pathologically.**
  These functions' `y` scales span roughly 1e-115 to 1e+3. Function 1 has
  values spanning dozens of orders of magnitude in the same dataset (down to
  `~1e-115` in collected data, with one dramatic outlier much larger than
  everything else — a "needle in a haystack" landscape, not a bug); function
  5's `y` runs up past 1000. This matters for `xi` (PI/EI), which is an
  **absolute** improvement margin in raw `y`-units, tuned by default assuming
  roughly unit-scale targets. At `y ~ 1e-16` a default `xi=0.01` completely
  swamps any real signal (verified: PI's acquisition value comes back as
  exactly `0.0` at the proposed point without correcting for this); at
  `y ~ 1e+3` the same `xi` is so small it does nothing at all.
  **Correct for this only when the data calls for it — see "When y-scaling is
  actually required" below. It is not a mandatory step for every notebook.**

## When y-scaling is actually required

`fit_y_scale`/`to_scaled_units`/`from_scaled_units` are a plain linear rescale
that exists to keep `xi` meaningful. They are **not** needed by default, and
adding them where they aren't needed just adds a units-mismatch hazard for no
benefit. Two verified facts narrow when it matters:

- **`fit_gp` passes `normalize_y=True`**, so the GP standardises `y` itself.
  Its posterior mean, posterior std, and ARD length-scales are identical
  (bit-for-bit, up to the rescale) whether you pass raw or scaled `y`. The
  surrogate never needs the rescale.
- **UCB is exactly scale-invariant, so `kappa` never needs it.** `UCB = mu +
  kappa*sigma` with `mu` and `sigma` both in `y`-units, making `kappa`
  dimensionless — verified to produce identical proposals on raw vs. scaled
  `y` for both `y ~ 1e-16` and `y ~ 1e+3` functions. (An earlier version of
  this file claimed `kappa` was absolute in `y`-units. That was wrong. The
  documented `0.0`-acquisition failure was PI, not UCB.) Note the *reported*
  acquisition value still scales linearly even though the argmax doesn't, so
  the `acq_value` column in `compute_iteration_diagnostics` is in whatever
  units you passed.

So: **scaling is only required when you're using PI or EI and the default
`xi` is mis-sized relative to the spread of `y`.** The check is one line —
`xi / (y.max() - y.min())` should land in roughly 0.1%–10%. Outside that,
either scale `y` or pick an `xi` sized for the raw data (both work; scaling
lets you keep using default-ish hyperparameters).

Measured on the initial batches, with `xi=0.01`:

| fn | D | y range | `xi`/spread | scaling for PI/EI? |
|----|---|---------|-------------|--------------------|
| 1 | 2 | -3.6e-3 … 7.7e-16 | 277% | moot — models `log10\|y\|`, see its own section |
| 2 | 2 | -0.066 … 0.61 | 1.5% | moot — uses UCB, `kappa` dimensionless |
| 3 | 3 | -0.40 … -0.035 | 2.7% | not needed |
| 4 | 4 | -32.6 … -4.0 | 0.035% | **required** |
| 5 | 4 | 0.11 … 1.1e+3 | 0.001% | moot — uses UCB, `kappa` dimensionless |
| 6 | 5 | -2.6 … -0.71 | 0.54% | not needed |
| 7 | 6 | 0.0027 … 1.36 | 0.73% | not needed |
| 8 | 8 | 5.6 … 9.6 | 0.25% | not needed |

Re-check as observations accumulate — a single needle-in-a-haystack result
can blow the range open (function 1's collected `y` reached `~-3.8e-115`).

Current state: all eight notebooks have an acquisition decision made from their own evidence.

Function 2 does not scale either, and since week 4 that is **structural rather
than a judgment to re-check**. It used to run EI, where `xi` is an absolute
margin in raw `y`-units and had to be re-verified against the `y` spread
whenever the data moved. It now runs UCB, and `kappa` is dimensionless — so no
movement in `y`, not even a function-1-style outlier blowing the range open,
can mis-size it. That closes the one live hazard the no-scaling decision
carried there.

**Function 1 no longer uses the linear rescale at all — it models
`log10|y|` instead.** An earlier version of this file said the rescale was
load-bearing there for the PI/EI rows of its backtest and for legible
printed numbers. The legibility part was true, but the framing was wrong:
`fit_y_scale` is a *linear* rescale, so on function 1 it maps
`[-3.6e-3, 7.7e-16]` to `[-3.48, 7.4e-13]` and leaves the **121 orders of
magnitude** of dynamic range completely intact. It cannot fix a
dynamic-range problem, only a units problem. See "Function 1: model
`log10|y|`, not `y`" below — that section supersedes this one for function 1.

## There is NO project-wide best approach — decide per function

This is the single most important thing to know before touching a notebook.
Each of the eight functions gets its own decisions about acquisition family,
hyperparameters, bounds, and y-scaling, made from that function's own
evidence. Do not propagate a choice from one notebook to another because it
worked there, and do not "tidy up" an apparent inconsistency between
notebooks — the differences are the findings.

The evidence for this is direct. The same diagnostics run on different
functions give genuinely opposed answers:

| fn | D | acquisition in use | why, in one line |
|----|---|--------------------|------------------|
| 1 | 2 | `ucb`, `kappa=3.0`, on a `log10\|y\|` surrogate | a GP on `y` is anti-predictive here (LOO R² −0.56; orders 42% of point pairs right, i.e. worse than chance); see below |
| 2 | 2 | `ucb`, `kappa=1.0`, x1 held at the incumbent | x1's length-scale is pinned, so an unrestricted search parks x1 on an edge at *any* `kappa`; holding x1 fixes that without paying `kappa`'s exploitation cost (see below) |
| 3 | 3 | `ucb`, `kappa=1.0`, restricted to x2 | only ONE input axis is informative; came off `max_variance` at week 5 once every x2 gap fell below the length-scale |
| 4 | 4 | `exploit`, week-2 outlier excluded from the fit | that one point doubled every length-scale and stalled `exploit` dead; EI is degenerate at *every* `xi` |
| 5 | 4 | `ucb`, `kappa=1.0`, x2+x3 searched, x0/x1 held | all 3 collected points were OUT OF DOMAIN and are excluded from the fit; the in-domain optimum looks to be on the x2=x3=1 boundary, where `kappa` makes no difference |
| 6 | 5 | `ucb`, `kappa=0.25`, all 5 axes live | no axis is pinned; `kappa` picked as the largest whose predicted mean still beats the incumbent |
| 7 | 6 | `exploit`, x1/x2 held at the incumbent | looks converged: NO `kappa` predicts a gain, so 0 is the value that loses least |
| 8 | 8 | `ucb`, `kappa=0.25`, x4/x7 held | `kappa` already correct by the predicted-gain rule; **this week spends the evaluation breaking a confound on x4** (see below) |

Functions 3 and 4 used to sit at opposite ends of the spectrum — `max_variance`
was function 3's choice and is still function 4's worst option by a wide margin.
Function 3 came off `max_variance` at week 5, which is worth reading as the
general lesson rather than as convergence: **the explore/exploit call is a weekly
decision on that function's current data, not a property of the function.** See
"Deciding when exploration is finished" below.

Three specific traps this rules out:

- **`kappa=5.0` is not a safe default, and the mode switch is not in the same
  place twice.** It was function 1's original choice (that notebook now uses
  `3.0`).

  **What a "mode switch" is.** `UCB(x) = mu(x) + kappa*sigma(x)`, so at any
  fixed `x` the score is a *straight line* in `kappa`. The acquisition surface
  typically has two local maxima: one inside the data (high `mu`, small
  `sigma`) and one far outside it (low `mu`, large `sigma`). The far mode's line
  is much steeper, so above some `kappa*` it overtakes the near one and the
  **argmax jumps discontinuously** — the acquisition *value* stays perfectly
  smooth, but the point returned does not. The crossover is
  `kappa* = (mu_near - mu_far) / (sigma_far - sigma_near)`.

  Measured on function 4: proposals move ~0.005 per step from `kappa` 1.0 to
  1.6, then **2.97 in one step** at 1.65, then nothing. Predicted crossover
  1.637, observed jump between 1.60 and 1.65.

  This matters more than ordinary saturation for three reasons: you cannot
  interpolate across it (`kappa=1.65` is not "slightly more exploratory" than
  1.6, it is a different question); the far side is usually a bounds corner
  where the GP has no data, so it is not more exploration but abandoning the
  surrogate; and it **moves as data arrives**, so a value that was safe last
  week may not be this week.

  Three functions have one, at different `kappa` and with different causes:
  - **Function 4**: now between `kappa=1.5` and `kappa=2`, having been between
    3 and 4 before the week-2 outlier was excluded from the fit — dropping a
    point moves the switch just as adding one does. Beyond it UCB proposes an
    extrapolated corner with predicted `y` about -37, in the region where the
    one real measurement came back at **-215**. Those coordinates sit on
    **upper** bounds, which are a `pad_fraction` artifact.
  - **Function 2**: at week 3, between `kappa=2` and `kappa=3`. Proposals
    travelled smoothly outward to `[0.683, 1.493]` at `kappa=2`, then snapped
    to `[0.922, 0.000]` — `x1` on its **lower** bound. That bound is
    `lower_limit=0.0`, a *real domain constraint*, not padding. Usable range
    at week 3 was `kappa <= 2`.

    **At week 4 that switch didn't just move, it reversed direction** — the
    strongest version yet of "not the same place twice." The new observation
    landed far out on x1 with an unremarkable `y`, so the GP's x1 length-scale
    flipped from a moderate `0.489` to pinned "irrelevant" at the upper
    `length_scale_bounds`. With x1 flattened, `kappa` 0 through 2 (the
    committed `1.0` included) land exactly on that same `x1=0` floor — even
    pure `exploit` hits it, so it isn't an explore/exploit tradeoff, it's the
    flattened mean drifting to an edge.

    **`KAPPA` was raised to `3.0` in response, and that was the wrong fix.**
    Recorded because the mistake is instructive: `kappa=3` proposes
    `[0.7327, 1.4528]`, and x1=1.4528 is *above* the highest observed x1
    (1.2951). It never escaped extrapolating on x1 — it swapped an excursion
    past a real domain boundary for one into `pad_fraction` padding, which by
    the per-axis rule below is the worse flavour. It also cost 0.14 of
    predicted mean (+0.442 vs +0.581) and 4.1 s.e. on the backtest.

    **The actual fix is to hold the pinned axis, not to raise `kappa`.** A
    pinned length-scale leaves the acquisition *flat* along that axis, so its
    argmax there is decided by wherever the optimiser stops — an edge — at any
    `kappa`. `kappa` is the wrong dial for that; it just changes which edge.
    Holding x1 at the incumbent and scanning x0 gives `[0.7175, 0.9266]`:
    nothing on a bound, nothing outside the observed range, inside the convex
    hull, and predicted mean +0.578. That is `function3.ipynb`'s construction,
    applied to the same cause. `KAPPA` is back to `1.0`. No real evaluation was
    ever spent at `3.0`.

    Caveat worth carrying: function 2's restriction is **weaker-warranted than
    function 3's**. Function 3's pinned axes are supported by data; here the
    low-x1 half of the observations has mean `y` +0.208 against +0.307 for the
    high half, and one of the best points sits at x1=0.124 with `y`=0.539. x1
    carries some signal, so freezing it gives up the chance to improve on it.
    Revisit the moment x1 comes off the pin — the proposal cell asserts on
    exactly that.

  - **Function 3**: between `kappa=1.0` and `kappa=1.5`, where the proposal
    switches x2 mode (0.868 -> 0.511) without touching any bound at all — so a
    switch need not involve a bound to matter. Its committed `KAPPA = 1.0` is
    the last step before the jump.

  So "a coordinate is on a bound" needs reading **per axis**, not as one
  failure mode: on an upper bound it usually means `pad_fraction` is choosing
  the point, while on the `0.0` floor it means the acquisition wants to leave
  the feasible region and cannot. Both are reasons to distrust the proposal,
  but they are not the same finding. `function2.ipynb`'s sweep cell prints
  bound contact and per-axis extrapolation as columns for exactly this reason.
  Note also that this sweep cell shares one random-number stream across all
  `kappa` rows, so on a multimodal acquisition surface its exact coordinates
  for a given row can lag a standalone call to `generate_next_point` — confirmed
  at week 4, where the sweep's `kappa=3` row (`[0.928, 1.099]`, UCB=0.995) was a
  worse local mode than the actual proposal cell's independent fit
  (`[0.733, 1.624]`, UCB=1.081). Trust the sweep for the *pattern*
  (wall vs. no wall); trust the proposal cell for the *coordinates*.
- **EI is not a safe default.** On function 4, `xi` swept from 0.5 to 20
  (0.24%–9.5% of the `y` spread) returns the *identical* corner every time —
  the improvement term is hopeless everywhere, so EI collapses onto its
  `sigma*phi(z)` term and silently becomes `max_variance`. No `xi`, in any
  unit system, fixes that. Check EI's actual proposal before assuming `xi`
  is a live dial.
- **The backtest must not be the decider.** `backtest_acquisitions` scores
  recognition of points whose `y` is already known, which is an exploitation
  task — it rewards ranking by posterior mean and gives no credit for
  reducing uncertainty. So it systematically favours `exploit` and low
  `kappa` and penalises `max_variance`, *whatever* the function. It ranked
  function 3's chosen config last. Use it as one weak input, and only where
  other evidence agrees. See the section on it below.

What to actually decide per function, each week:

1. **Acquisition family and hyperparameter** — from `compare_kappa_proposals` /
   `compare_xi_proposals`, plus inspecting where each candidate family would
   actually propose. Watch for discontinuities, not just saturation.
2. **Bounds** — `pad_fraction` is a choice. The standard `1.0` doubles each
   axis, which on function 3 left ~90% of the box as extrapolation and
   produced a corner proposal; that notebook uses the observed bounding box
   instead. See "Checking a proposal is not an extrapolation" below.
3. **y-scaling** — only when required; see the section above.
4. **Whether the GP is even trustworthy** — `get_length_scales`,
   `loo_predictions`, and a refit without the most extreme point. A single
   observation reshaping the kernel is a reason to distrust variance-driven
   acquisitions.

## Function 1: model `log10|y|`, not `y`

Function 1 is the one function where the *modelling target itself* is a
decision, not just the acquisition. Verified on 12 observations, all under
that notebook's box bounds:

| target | LOO R² | pairs ordered right | mean \|z\| | ARD length-scales |
|---|---|---|---|---|
| `y`, linearly rescaled | **−0.56** | **42.4%** (28/66) | **1.3e12** | `[0.027, 10.0]` — x1 pinned |
| `log10\|y\|` | **+0.68** | **86.4%** (57/66) | **0.80** | `[0.239, 0.274]` — neither pinned |
| `sign(y)` as ±1 | −0.82 | 6.2% (2/32) | 1.91 | `[10.0, 0.058]` — x0 pinned |

"pairs ordered right" is a plain count, not a rank statistic: over every pair of
observations, how often does the surrogate put them in the same order as
reality? **50% is a coin flip.** It is the property an acquisition function
actually depends on, since it only ever *compares* candidate scores to take an
argmax — it never uses the posterior mean's absolute value. Ties are skipped,
which is why the `sign(y)` row has only 32 pairs (the opposite-sign ones).
R² is scale-free so it is comparable across targets; RMSE/MAE would not be.

A GP on `y` here is **anti-predictive**: R² below zero means worse than
predicting the mean, and it orders only 42% of point pairs correctly — worse
than the 50% a coin flip would manage. Ordering is the only use an acquisition
function makes of the posterior mean.

The mechanism is worth knowing exactly, because it is what tells you a
*linear* rescale can never fix it. `fit_gp` passes `normalize_y=True`, and
standardising function 1's `y` gives the GP this:

```
[-3.316625, 0.301511, 0.301511, ... 0.301511]   # eleven identical values
```

Those eleven agree to 13 significant figures. Their whole spread is 7.7e-13
against a fitted `WhiteKernel` noise std of 1e-3 — **7.7e-10 of the GP's own
noise floor** — so its noise model declares them one measurement repeated at
eleven locations. The surrogate therefore had **2 distinct y-values, not 12**,
and fit the only thing two values describe: a step. It cannot tell the
incumbent best (7.7e-16) from the worst point on record (3.3e-124) because in
its working units they are the same number. A linear rescale preserves every
ratio, so it leaves that untouched; `|y|` spanning 121 orders of magnitude is
a dynamic-range problem, not a units problem.

Caveat on that table's `mean |z|`: 1.3e12 is carried by **one point** (median
|z| is 0.25). Holding out the outlier leaves eleven identical values, so the
GP infers a constant function, reports a LOO std of 2.3e-13, and is then wrong
by 3.6 standardised units. The error bars are not uniformly too tight — the
one informative observation is simply unpredictable from the others and the GP
doesn't know it. Read median alongside mean here, as with backtest regret.

Three things to keep straight about this:

- **It is not the removed `log1p` transform.** That one assumed `y >= 0` and
  was correctly deleted when that turned out false. This one models `|y|` and
  treats sign as a separate question, so it assumes nothing about `y`'s sign.
  It does need a floor (`LOG_FLOOR = -130.0`) because `log10(0)` is undefined.
- **The objective still decomposes exactly**: `max y` is the largest `|y|`
  *among points where `sign(y) > 0`*. So `log10|y|` is only the magnitude
  half. Sign is **not yet learnable** (R² −0.82 on 4 negatives in 12 points),
  which is why that notebook measures the sign rather than modelling it.
- **The log-magnitude GP over-predicts near zero contours.** `|y| → 0` at a
  sign change, so `log10|y| → −∞`, and a smooth GP with no data in the gap
  cannot represent that dip. Predicted magnitudes *inside* a sign bracket are
  **upper bounds, not estimates**.

**The bounds change is coupled to this and is not optional.** Under
`pad_fraction=1.0` the data covers only 22.7% of function 1's box. The old
broken surrogate hid that, because its pinned x1 length-scale flattened the
posterior std. Once the GP actually fits, σ is largest in the empty padding
and log-magnitude UCB proposes `[1.6853, 1.6811]` — the bounds corner, pure
extrapolation. Under the observed bounding box it proposes `[0.5339, 0.4936]`,
inside the data. Fixing the target without tightening the box would have made
that notebook worse.

**The live limitation: UCB on `log10|y|` is sign-blind, and that makes
`kappa` load-bearing.** The surrogate models magnitude, so UCB maximises
`|y|` — but the objective is `max y`, and function 1's largest `|y|` on record
is **negative** (`−3.61e-03` at (0.650114, 0.681526), 13 orders above anything
else). The surrogate's own peak is therefore somewhere we do *not* want to go,
and UCB cannot tell. Measured on the `log10|y|` fit under box bounds:

| `kappa` | proposal | distance to the negative peak |
|---|---|---|
| 0.5 | `[0.6426, 0.6086]` | **0.073** — effectively resampling it |
| 1.0 | `[0.6512, 0.5900]` | 0.092 |
| 2.0 | `[0.6673, 0.5618]` | 0.121 |
| 3.0 | `[0.6770, 0.5407]` | 0.143 |
| ≥ 4.0 | `[0.0825, 0.8799]` | 0.601 — **both axes on the box edge** |

(Re-measured after the week-4 observation at `[0.53388, 0.493601]`, `y =
8.27e-14`, landed almost exactly on the prior week's proposal. That single
point moved the switch from `kappa 8 → 10` to `kappa 3 → 4`, and pulled
`kappa=3`'s own proposal from 0.221 to 0.143 from the negative peak — closer
to the wrong-sign region than before, not farther. The window did not just
shrink at the edges, it shrank at `KAPPA`'s own value.)

So the usable window is bounded on *both* sides: low `kappa` exploits into the
wrong-sign peak, `kappa >= 4` now collapses onto a box corner, and `KAPPA =
3.0` sits at the very last usable step before that switch, not comfortably
inside it. Re-run this sweep every week — the margin here has already
evaporated once from a single observation and there's no reason to assume it
won't move again. Note this is a different failure mode from function 4's
switch — same lesson (watch for discontinuities, not just saturation),
different cause.

**Sign stays unmodelled**, so record it every week; it is half the picture and
nothing in the notebook predicts it. Related geometry worth re-reading each
week: the incumbent and the magnitude peak are only 0.0959 apart with opposite
signs, so a zero contour runs between them and `max y` sits just on its
positive side. The incumbent is a genuine local max rather than a near-zero
point (its LOO z on `log10|y|` is +0.18, right on the trend). The notebook
reports the sign brackets as a diagnostic, but nothing acts on them — the
proposal is `generate_next_point`'s UCB argmax.

## A frozen axis confounds itself: function 8

Holding an axis at the incumbent is a useful construction (functions 2, 3, 5, 7
and 8 all do it), but it has a cost that took until week 5 to notice: **an axis
frozen at one value across every collected point cannot have its effect
separated from whatever else those points share.**

Function 8 is the worked example. Its three collected observations are the three
highest `y` values on record, and all three sit at exactly `x4 = 0.586936` —
the only observations at that value, frozen there since week 2:

```
rank   src         y       x4       x7
   1   wk4    9.9456   0.5869   0.6031
   2   wk2    9.9280   0.5869   0.6031
   3   wk3    9.8021   0.5869   0.6031
   4  init    9.5985   0.4039   0.8931
```

With x4's length-scale **pinned at the 10.0 ceiling** — the GP's own statement
that it found no structure there — it fits the smoothest thing consistent with
that cluster, a linear ramp, and extrapolates it to the domain boundary.
Including x4 in the search then claims a **30x larger predicted gain** (+0.185
against +0.006). The raw data disagrees: Spearman(x4, y) is **−0.120** on the
initial 40 points alone, the five highest-x4 initial observations give
y = 8.54, 8.16, **5.84**, 6.45, 6.89, and the high-x4 half averages 0.190 *below*
the low half.

So the GP claims x4 increases `y` while its own length-scale reports no
structure and the data trends the other way. That is a confound, not a finding.

**How to spot it:** for each axis, check whether `new_X[:, d]` takes a single
value. If it does and the collected points are also the best ones, any GP
direction on that axis is suspect. `function8.ipynb`'s axis cell now prints this
automatically.

**How to break it:** one controlled evaluation — move the frozen axis and change
*nothing else*. Because only that coordinate differs from the incumbent, the
result compared against the incumbent's measured `y` isolates the axis's effect,
and the two hypotheses make opposite predictions. Choose the value where they
diverge most (function 8 uses x4 = 1.0, the GP's own argmax).

Two cautions. This spends a week on a diagnostic rather than an improvement, so
it is worth it only when the claimed gain is large relative to the safe
proposal's — function 8's +0.185 vs +0.006 clears that bar, and its x7 (+0.010)
does not, so x7 stays confounded for now. And **do not confuse this with
function 7's case**: there the flat axes x1/x2 were swept by the 30-point initial
design, three separate tests agreed they were irrelevant, and testing them would
have been waste. The distinguishing question is not "is the axis held?" but "is
it held at a single value *in the collected points that also happen to be the
best ones*?"

## Choosing `kappa`: the predicted-gain criterion

Functions 6 and 7 produced a replacement for the criterion those notebooks used
to carry — *"the largest `kappa` whose proposal touches no upper bound"*. That
one rested on upper-bound contact meaning `pad_fraction` was choosing the point,
which stopped being true when bounds became the true domain `[0,1]`.

**The replacement: pick the largest `kappa` whose predicted mean still beats the
incumbent.** Compute the restricted proposal at several `kappa`, read off its
posterior mean, and compare to `y.max()`. It is checkable, it is printed each
run, and it degrades gracefully.

The same criterion gives opposite answers on the two functions, which is the
point of applying it per function rather than inheriting a value:

| fn | kappa=0 | 0.25 | 0.5 | 1.0 | committed |
|----|---------|------|-----|-----|-----------|
| 6 | **+0.0043** | **+0.0029** | −0.0034 | −0.0489 | `0.25` — largest with a gain |
| 7 | −0.0031 | −0.0043 | −0.0086 | −0.0276 | `0` (`exploit`) — none gains |

On function 6 two values still predict an improvement, so the largest of them
wins. On function 7 **nothing does**, so `kappa=0` is chosen as the value that
loses *least* — and that is itself the signal that the function is converged
locally, not a reason to keep shrinking `kappa`.

Two things learned alongside it:

- **A long length-scale is not "no effect".** Selecting axes to hold by
  `length_scale >= domain_width` conflates *no short-range structure* with *no
  influence*. On function 6, x4's length-scale is 1.225 (above the domain width
  of 1.0) yet it moves the posterior mean by 0.958 — nearly as much as the most
  informative axis — because it is **monotone** across the domain. Select held
  axes by GP mean-variation (function 6 and 7 use "< 5% of the largest"), and
  keep the genuinely-pinned check (`length_scale` at the 10.0 ceiling) as the
  separate, stronger signal it is.
- **A monotone axis will sit on a domain bound, and that is correct.** Function
  6's x4 has Spearman −0.654 and its proposals land at x4=0 every time. Both
  notebooks now print, before the proposal, which axes are monotone and which
  end is their best, so an edge contact can be read as *expected* rather than as
  the optimiser stopping arbitrarily.

**Compute gap ratios on in-domain points only.** An out-of-domain observation
inflates the apparent gap: on function 7 the all-points ratio for x3 is 1.85 and
looks unresolved, but the gap runs from x3=0.961 to the out-of-domain 1.668, so
most of it is unreachable. In-domain the ratio is 0.23.

## Deciding when exploration is finished

Function 3 produced a clean, reusable criterion for the explore/exploit call.
Prefer it to arguing the question in the abstract.

**Compare each gap between neighbouring observations, along the axis that
matters, against the fitted length-scale for that axis.** Convert the
length-scale out of normalised units first: `get_length_scales(gp)[d] *
(bounds[d,1] - bounds[d,0])`. A gap *wider* than the length-scale is one the GP
cannot interpolate — that is where hidden structure survives. Gaps below it are
interpolable, and sampling them mostly confirms what the surrogate already
believes.

The evidence on function 3:

| week | largest x2 gap | x2 length-scale | ratio | outcome |
|---|---|---|---|---|
| 4 | 0.149 | 0.110 | **1.35** | sampled it; came back ~0.08 **above both** brackets |
| 5 | 0.109 | 0.121 | **0.90** | switched to `ucb`/`kappa=1.0` |

Two cautions learned the hard way there:

- **Bracket agreement is not evidence that the interior is boring.** At week 4 the
  two observations bracketing the proposal differed by only 0.004, and that
  notebook's header called it "a weaker bet" on those grounds. That was wrong: the
  gap exceeded the length-scale and a sharp peak was sitting inside it. Use the
  gap/length-scale ratio, not the bracket values.
- **Do not dismiss a posterior-mean bump as an artifact without this test.**
  When the GP's fitted amplitude greatly exceeds the data's spread, its
  posterior mean *overshoots* between sparse observations — it rises above both
  bracketing values, and can exceed the best value ever measured. On function 3
  the kernel amplitude is 1.647 against a `y` std of 0.079, a ratio of **20.8**,
  and the mean overshoots in 5 of the gaps along x2.

  An earlier version of `function3.ipynb` called this "ringing" and rejected EI
  on that basis: EI wanted x2 ≈ 0.858, where the mean bumped to about −0.044
  between observations at 0.771 (`y = −0.118`) and 0.920 (`y = −0.114`).
  **That dismissal was wrong.** Week 4 sampled x2 = 0.8438, 0.014 away, and it
  returned **−0.0356** — better than the GP had predicted and ~0.08 above both
  brackets. The overshoot was real structure. `max_variance` reached nearly the
  same place two weeks later for an unrelated reason (widest gap); EI would have
  got there sooner.

  Two things follow. A large amplitude-to-spread ratio tells you the GP *can*
  overshoot, **not that a given bump is spurious** — that needs a separate test,
  and the gap/length-scale ratio is the one that worked here (0.149 vs 0.110,
  i.e. genuinely unresolved, exactly where the peak was). And "ringing" was the
  wrong word regardless: profiling the gap shows a single smooth hump, not an
  oscillating train, so **"overshoot between sparse observations"** is the term
  used now.

- **The ratio rests on a fitted length-scale, which can be optimistic.** Function
  3's GP believed a 0.149 gap was interpolable and it was not. So "no gap exceeds
  the length-scale" is a reason to *lean* toward exploitation, not proof that
  exploration is done. If the resulting exploit proposal misses, read that as
  evidence to go back to exploring rather than to exploit harder.

One structural trap this also exposes: **a variance-driven acquisition can never
question a pinned dimension.** On function 3 the posterior std moves only
0.0143 -> 0.0180 across the entire x0/x1 range, so `max_variance` would never
propose varying them however long it ran.

The companion rule, learned on function 2: **when an axis is pinned, hold it —
don't reach for `kappa`.** A pinned length-scale makes the acquisition flat along
that axis, so its argmax there is set by wherever the optimiser stops, which is
an edge, at *every* `kappa`. Changing `kappa` only changes which edge, while
paying the usual exploration/exploitation cost for nothing. Both functions 2 and
3 now hold their pinned axes at the incumbent and scan only the informative ones,
and both assert that the axis is still pinned so the construction fails loudly
rather than silently becoming invalid. Testing a pinned axis has to be a
deliberate choice. Before making it, check the data directly — on function 3,
pairs of observations with similar x2 differ no more in `y` when x0/x1 are far
apart (mean |dy| 0.0159, n=9) than when they are close (0.0202, n=5), against an
overall `y` std of 0.0793. That supports the pinning independently of the GP,
which matters because it could otherwise be self-reinforcing: all three collected
points on that function sit on a single x0/x1 line.

## Excluding an observation from the fit

Function 4 is the one case where a real measurement is **kept in the record but
left out of the GP fit**. Done via `EXCLUDE_FROM_FIT = [0]` in its data cell,
which splits `X_all`/`y_all` (the full record, for reporting) from `X`/`y` (the
modelling set, used by everything else). Setting it to `[]` reverts completely.

**The bar it had to clear.** Not "this value is inconvenient" — the test was
whether one observation was reshaping the whole surrogate:

| | x0 | x1 | x2 | x3 | y std |
|---|---|---|---|---|---|
| excluded | 0.860 | 0.853 | 0.890 | 0.803 | 8.00 |
| included | 1.849 | 1.986 | 1.621 | 1.690 | 34.99 |
| inflation | 2.15x | 2.33x | 1.82x | 2.11x | 4.4x |

Doubling every ARD length-scale over-smooths the posterior, which suppressed
sigma exactly where the search needed resolution. `exploit` had stalled to a
step of 0.0021 with a predicted gain of +0.0011; with the point excluded the same
acquisition steps 0.0333 and predicts +0.0828.

**Why this one qualifies.** It was a bad *submission*, not a bad function value:
the old notebook proposed an extrapolated corner with two negative coordinates
(bounds built without `lower_limit=0.0`), and the result, -215, is 6.6x worse
than anything else. It measures a region since ruled out of the search.

**Three cautions before doing this anywhere else.**

- **It is not a licence to drop awkward data.** CLAUDE.md's standing advice is
  that a refit without the most extreme point is a *distrust signal* (see
  **Model diagnostics**). Excluding it is a bigger step and needs the
  length-scale evidence above, not just a large residual.
- **The surrogate stops knowing that region is bad.** Nothing in the model then
  prevents the search returning there. On function 4 the **convex-hull check** is
  what stops it — dropping the point shrinks the hull, so a corner proposal fails
  it. Whatever the function, identify the guard explicitly before excluding.
- **Recheck everything indexed off the collected rows.** The exclusion silently
  broke two things in that notebook: a LOO check that identified the outlier by
  `worst == n_initial` (that index is now week 3), and an iteration count using
  `len(new_y)` where the replay only sees `len(y) - n_initial` rows. Both were
  off-by-one against the modelling set rather than the record.

## Checking a proposal is not an extrapolation

Two different tests, and the distinction matters:

- **Per-axis** — is each coordinate within the range already observed on its
  own axis? This is what box bounds guarantee. It is the weak test.
- **Convex hull** — is the point inside the smallest convex set containing
  the observations? Strictly stronger, and the meaningful one.

A point can pass the per-axis test and fail the hull test: a corner of the
bounding box can sit far outside the data cloud. This is not hypothetical —
on both function 3 and function 4, rejected proposals passed per-axis and
failed the hull check. Bounding box volumes overstate the data's reach badly
in higher D (function 3: hull is 34.5% of the box; function 4: 11.5%).

`initial_bounds` / `fit_gp` / `generate_next_point` only accept a `(D, 2)`
box, so a hull check can only ever be a post-hoc validation of the resulting
point, not a constraint on the search:

```python
from scipy.spatial import Delaunay
hull = Delaunay(X)
in_hull = bool(hull.find_simplex(x_next) >= 0)
```

`function3.ipynb` and `function4.ipynb` both run this on their proposals.

## Standard per-function weekly workflow

The *mechanics* below are shared; the *choices* inside them are per-function
(see above).

**Collected observations live in `weekly_data/function_N/`** as two flat CSVs.
The initial batch stays in `initial_data/function_N/*.npy` and is never touched.

```
weekly_data/function_N/inputs.csv    x0..x{D-1}, one row per point proposed, 6 dp
weekly_data/function_N/outputs.csv   y,          one row per result returned
```

**There is deliberately no week column.** Alignment is positional and the
invariant is `len(inputs) - len(outputs) in (0, 1)`: 0 means every proposal has
a result, 1 means the last input row is this week's proposal awaiting its result
("pending"). `load_collected` enforces that and returns the pending row
separately, so it is never fed to the model. A larger gap means a week was
skipped or `record_results.py` was not run, and it raises.

**Both writers REWRITE rather than append**, which is what makes the whole thing
idempotent:

- `save_proposal(N, new_X, x_next)` — the last cell of each notebook — writes the
  resulted rows plus the current proposal. Re-running a notebook any number of
  times leaves exactly one row per week; verified byte-identical over four
  consecutive runs. A previous pending row is *replaced*, which is what you want
  when revising a proposal before submitting.
- `record_results.py` rewrites all eight `outputs.csv` from **all** lines of
  `outputs.txt`, which stays the source of truth. Running it twice is a no-op.

This does re-introduce something previously removed — an earlier file-based
history log with pending-proposal tracking was built and then deleted in favour
of in-memory arrays. This version is deliberately smaller: two flat CSVs holding
*only* observations, no proposal metadata or diagnostics, and the notebooks keep
their explicit structure rather than being wrapped in `propose_and_log` /
`record_observation` helpers. What it buys is transcription safety — the two
data-entry bugs this project has had (function 4's sign flip, function 5's
6-dp rounding overshoot) both came from hand-copying between notebooks and the
text files.

Coordinates are stored at **6 dp**, matching what is actually submitted and
evaluated. That is safe now in a way it was not before: 6-dp rounding of any
value inside `[0,1]` stays inside `[0,1]`, so the failure that hit function 5
(a padded bound at 1.670021516812 with a submission of 1.670022) cannot recur
under unit-cube bounds.

**Row order is chronological and load-bearing.** `compute_iteration_diagnostics`
replays it, and `EXCLUDE_FROM_FIT` indexes into it positionally (functions 4 and
5). Never reorder rows.

### The weekly cycle

```
1.  python record_results.py          # outputs.txt -> the 8 outputs.csv; gap goes 1 -> 0
2.  run functionN.ipynb               # loads the store, proposes, rewrites inputs.csv; gap 0 -> 1
3.  submit the printed 6-dp string    # identical to the row just written
```

Step 1 first, always. If you submit a proposal, get the result, and re-run the
notebook *before* syncing, the rewrite replaces the pending row and the record
of what you actually submitted is lost. `load_collected` prints a warning
whenever it is about to replace a pending row.

`python record_results.py --check` validates alignment across all eight without
writing. `--bootstrap` rebuilds `inputs.csv` from `inputs.txt` as well, which is
the one-time migration and should not normally be needed again.

```python
from bayes_tools import (
    initial_bounds, validate_bounds_consistency, append_observations,
    fit_y_scale, to_scaled_units, from_scaled_units,  # only if scaling is required
    generate_next_point, ucb_acquisition, fit_gp, get_length_scales,
    loo_predictions, compare_kappa_proposals, print_kappa_comparison,
    backtest_acquisitions, print_backtest_summary,
    compute_iteration_diagnostics,
)
from viz_tools import (
    plot_2d_bo, plot_nd_slices, plot_loo_calibration,
    plot_kappa_sensitivity, plot_acquisition_backtest, plot_bo_diagnostics,
)

# Loaded ONCE, never modified again.
X_initial = np.load("initial_data/function_N/initial_inputs.npy")
y_initial = np.load("initial_data/function_N/initial_outputs.npy")
n_initial = len(y_initial)
D = X_initial.shape[1]

# Collected observations, loaded from weekly_data/function_N/.
# pending_X holds 0 or 1 rows -- a proposal with no result yet -- and is NOT
# passed to the model.
new_X, new_y, pending_X = load_collected(N, D)
```

Each week, **in this order** (append before propose — an earlier draft got
this backwards and it matters: you're recording last week's result before
asking the next question, not the other way round):

```python
# 1. Results are already in place -- `python record_results.py` did that from
#    outputs.txt before this notebook was opened. Nothing to append by hand.

# 2. Combine initial + everything collected so far.
X, y = append_observations(X_initial, y_initial, new_X, new_y)
bounds = initial_bounds(X_initial, pad_fraction=1.0, lower_limit=0.0)
validate_bounds_consistency(X, bounds)

# 3. ONLY IF REQUIRED (PI/EI with a mis-sized default xi — see the table above;
#    skip this entirely for UCB, and for any function whose y is near unit scale).
#    If you do scale, re-fit every week — don't assume last week's y_scale still fits.
y_scale = fit_y_scale(y)
y_scaled = to_scaled_units(y, y_scale)

# 4. Propose the next point. Pass y_scaled if step 3 applied, plain y otherwise.
#    The acquisition and hyperparameter here are PER-FUNCTION -- there is no
#    default. See the per-function table above. Note function 1 doesn't even
#    pass `y` here any more; it passes log10|y| (see its own section above).
x_next, gp = generate_next_point(X, y, bounds, acquisition=..., maximize=True, ...)
print("Evaluate this point:", x_next)

# 5. Record the proposal (last cell of every notebook). Rewrites inputs.csv
#    as the resulted rows plus this proposal, so re-runs cannot duplicate.
save_proposal(N, new_X, x_next)
```

**Critical consistency rule**: pick one unit system per session and use it for
*everything* — `generate_next_point`, `compare_kappa_proposals`/
`compare_xi_proposals`, `backtest_acquisitions`,
`compute_iteration_diagnostics`, `loo_predictions`, and the plots. Mixing raw
and scaled `y` in the same session silently corrupts everything downstream.
If you did scale, call `from_scaled_units` only at the very end, for numbers
you're about to print or plot. If you didn't scale, don't import the
`*_scaled_units` helpers at all — that way there's nothing to mix up.

A useful pattern for the acquisition hyperparameter itself: bind it to one
module-level constant (e.g. `KAPPA = 1.0`, `XI = 0.01`) and pass that same
constant to the proposal, the plot's `acq_kwargs`, and the diagnostics replay,
so they can't silently drift apart. **Functions 1 (`KAPPA = 3.0`) and 2
(`KAPPA = 1.0`) do this.** (`exploit` and `max_variance` take no hyperparameter
at all, so functions 3 and 4 have nothing to bind — one less thing to
desynchronise.)

## Before committing to acquisition hyperparameters (no real evaluations spent)

Real evaluations are precious, so don't just guess `kappa`/`xi`/acquisition
family — check first, using only data already collected:

- **`compare_kappa_proposals`** / **`compare_xi_proposals`** (+ `print_*_comparison`,
  `plot_kappa_sensitivity`/`plot_xi_sensitivity`): fits ONE shared GP, varies
  only the hyperparameter, shows where each choice would propose to search
  *right now*. Read it by watching whether `x_next`/`distance_from_incumbent`
  keeps changing as the hyperparameter increases — once proposals stop
  moving, you've saturated exploration and larger values buy nothing.
  **Also watch for a discontinuity**, which matters more than saturation:
  function 4's proposals are near-identical for `kappa` 0.5–3 and then jump
  to a bounds corner at `kappa >= 4`. And check the hyperparameter is a live
  dial at all — on function 4, EI returns the same point for every `xi` from
  0.5 to 20.
- **`backtest_acquisitions`** (+ `print_backtest_summary`, `plot_acquisition_backtest`):
  repeatedly splits existing data into a seed/candidate split, checks which
  acquisition config best recognizes good points already in the dataset.
  **Structural limitation, keep this in mind**: this can only test
  recognition of already-known points — it cannot test which config is best
  at proposing a genuinely new location, since there's no ground truth for
  that without a real evaluation. Treat results as directional, not proof.
  **It is also biased, not just limited**: scoring recognition is an
  exploitation task, so it rewards ranking by posterior mean and gives no
  credit for reducing uncertainty. It therefore favours `exploit` and low
  `kappa` and penalises `max_variance` on every function, regardless of what
  is actually appropriate — it ranked function 3's chosen config last, and
  "`exploit` wins the backtest" is close to tautological. Two further
  cautions: rankings among the top few configs are often within noise and
  have flipped on the addition of a single observation (function 2), and mean
  regret is inflated by extreme outliers sitting in the candidate pool, so
  read median regret too (function 4).

  **Function 2 is the worked example of that instability.** Its ranking changed
  at all three sample sizes: at 10 points `ucb_k1` led with `ei` third; at 11
  points `ei` led a four-way dead heat; at 12 points the order is `ucb_k1`
  0.165, `exploit` 0.180, `pi` 0.198, `ei` 0.264, with `ei` for the first time
  *separably* behind the leader (+0.099 ± 0.030 paired over the same 50 splits,
  3.3 s.e.). The direction it moved is exactly the one the metric's bias
  predicts, toward `ucb_k1`/`exploit`.

  **That notebook switched to `ucb`/`kappa=1.0` at week 4 anyway** — a
  deliberate decision taken with the bias above in view, not a reading of the
  table at face value. Two things are worth recording about it:

  - One detail cuts against the pure-bias story: `ucb_k1` (0.165) beats
    `exploit` (0.180), so the metric is not simply rewarding whatever is
    greediest.
  - But `ucb_k1`, `exploit` and `ucb_k0.5` are all within noise of each other,
    so what the backtest actually supports is **"`kappa` at or below 1"**, not
    `kappa=1` specifically. The firmer constraint on that notebook is the
    `kappa=2 -> 3` mode switch described above.

  The trade being made is explicit: `kappa=1` is a step toward exploitation, and
  EI's variance-seeking behaviour on this data was the original argument for EI.
  If a week-5 result comes back below the incumbent, that is the first thing to
  re-examine.

  **Week 4 cuts both ways, and the second half is the more useful lesson.**
  The backtest put `ucb_k0.5` (0.159) and `ucb_k1` (0.165) ahead of everything
  else, so reading the table alone would have kept `KAPPA=1.0`. The direct
  bound-contact check showed something the backtest cannot see — `kappa` 0–2 all
  land on x1's real floor — because it only ever scores recognition of
  already-known points and never asks whether a proposal is pressed against a
  domain wall. That much is a genuine win for "the backtest must not be the
  decider": a real defect, invisible to the metric.

  **But overriding the backtest is not the same as being right.** `KAPPA` was
  raised to `3.0` on that evidence and it did not fix the wall — see the
  function-2 entry above. The backtest's objection (a 4.1 s.e. penalty at
  `kappa=3`) turned out to be pointing at a real cost, even though it could not
  articulate the reason. The lesson is narrower than "ignore the backtest when
  it conflicts with a structural check": when the two disagree, **look for a
  third option that satisfies both** before paying either price. Here that
  option existed — hold the pinned axis, keep `kappa=1` — and it took a week to
  spot because the disagreement was framed as a choice between two `kappa`
  values.

  Compare configs **paired** where you can: every config is scored on the same
  splits, so the per-split regret difference and its standard error tell you
  whether a gap is real, which the two means side by side do not.
  `function2.ipynb`'s verdict cell does this and needs no statistical test —
  just the mean and standard error of the paired differences.

## Model diagnostics

- **`get_length_scales(gp)`** — `fit_gp` uses ARD (one Matern length-scale
  per input dimension, not a single shared value). Short length-scale =
  that dimension matters; pinned at the upper `length_scale_bounds` = the
  GP thinks that axis is close to irrelevant. Don't trust relative
  magnitude between two dimensions that are both pinned at the bound — only
  trust "pinned vs. not pinned."
- **`loo_predictions` + `plot_loo_calibration`** — leave-one-out calibration
  check. Refits the GP once per point, leaving it out, predicts it from the
  rest. This is the main way to validate the surrogate once D is too large
  to eyeball the fitted surface directly (roughly 4D+), or on a function
  whose y-scale is extreme (1, 4, 5) where you want to confirm the GP is
  learning something real rather than fitting noise.
- **`compute_iteration_diagnostics(X, y, bounds, n_initial, acquisition=, kappa=/xi=, maximize=)`**
  — reconstructs per-iteration history (acquisition value, GP hyperparameters,
  domain uncertainty at each past proposal) purely by replaying ordered
  `X`/`y` — no persisted log. **Must be called with the SAME acquisition
  settings that were actually used to generate each historical proposal** —
  this was a real bug caught mid-project (diagnostics were replayed with
  `pi`/`xi=0.01` while actual proposals used `ucb`/`kappa=5.0`; the resulting
  "acquisition value" was meaningless). Also assumes row order in `X`/`y` is
  true chronological order — never reorder rows after `append_observations`.
- **`domain_grid_n` MUST be lowered past D=4, or the kernel dies.**
  `compute_iteration_diagnostics` computes its `domain_mean_std` field by
  averaging posterior std over a dense grid of `domain_grid_n ** D` points, and
  the default `domain_grid_n=40` is exponential in `D`:

  | D | grid points at the default | array size |
  |---|---|---|
  | 4 (functions 4, 5) | 2.6e6 | 0.08 GB — fine |
  | 5 (function 6) | **1.0e8** | **4.1 GB — kills the kernel** |
  | 6 (function 7) | 4.1e9 | 197 GB |
  | 8 (function 8) | 6.6e12 | 419,000 GB |

  Function 6's first run died exactly there. Functions 6, 7 and 8 therefore size
  it as `DOMAIN_GRID_N = max(3, int(200_000 ** (1.0 / D)))`, holding the grid
  near 200k points whatever `D` is. The cost is that `domain_mean_std` becomes a
  coarser estimate: still a fair basis for comparing one iteration against
  another *within* a run, but its absolute value is **not** comparable across
  runs that used different grids. Nothing else in the returned history depends
  on the grid.
- **Never pass `y_scale` to a plot when the GP was fit on a non-linear target.**
  It is a *linear* rescale correction, meaningful only for a GP fit on
  `to_scaled_units(y, y_scale)`. Applying it to function 1's `log10|y|` surface
  would be a units error, so that notebook passes `y_scale=None` and labels its
  panels in log units.
- `plot_bo_diagnostics`, `plot_convergence`, `plot_acquisition_decay`,
  `plot_uncertainty_shrinkage`, `plot_step_distance` do **not** have
  `y_scale` support yet — their axis numbers are in whatever units of `y`
  you passed in. `plot_sample_trajectory`/`plot_bo_diagnostics` are 2D-only
  (hardcode a 2D scatter panel) — for D>2, use `plot_nd_slices` instead of
  `plot_2d_bo`, and skip the trajectory panel.

## Keeping sklearn's warnings out of printed tables: `fitting()`

`bayes_tools.fitting(label)` is a context manager. Wrap every GP fit in a cell
in it, then print afterwards:

```python
from bayes_tools import fitting

with fitting("leave-one-out over 3 candidate targets"):
    ...          # all the GP fitting for this cell
...              # then print freely; no warning can interrupt it
```

**The problem it solves.** Every fit can raise a burst of sklearn
`ConvergenceWarning`s saying a hyperparameter is pinned at a bound. They go to
**stderr**, so if a cell interleaves fit/print/fit/print they land between the
printed rows and make a table unreadable. `loo_predictions` refits once per
point, so one call can emit dozens; a 50-split `backtest_acquisitions` emits
more. Per full run: function 1 ~88, function 2 ~99, function 3 ~152,
function 4 ~110, function 5 ~188, function 6 ~172, function 7 ~296,
function 8 ~339 — the count grows with D and with how many refits a
notebook does.

**It reports, it does not suppress.** Warnings are captured for the duration of
the block, then printed once, collapsed to one line per distinct message with a
count — `39 x ConvergenceWarning k2__noise_level (dim 0) pinned at lower bound
1e-06` rather than 39 identical stderr dumps. Totals are shown in full and the
fit is unaffected; only where the messages go changes. Silent if nothing was
raised.

**Why not just filter them.** What they report is real evidence: a pinned
length-scale means the GP considers that axis close to irrelevant (see **Model
diagnostics** above), and on function 1 that verdict is part of how the
modelling target was chosen. Deduplicating keeps the signal and drops only the
repetition. Note too that raising `length_scale_bounds` does not make the
warning go away in any useful sense — a pinned length-scale says "irrelevant"
just as clearly at 100 as at 10 (see `function3.ipynb`'s header).

**Two rules for using it.**

1. Put *all* of a cell's fitting inside the block and do the printing after it.
   A fit that runs after a `print` will still interleave, `fitting()` or not.
   Functions 1 and 2 had three such cells between them; each was fixed by
   hoisting the fit to the top of the cell.
2. Import it, don't paste it. It lived as a copy-pasted block in `function1`
   and `function2` first and had already drifted between them; it now lives
   only in `bayes_tools.py`. **All eight notebooks import it**; none defines
   its own copy.

## Visualization by dimensionality

- **1D**: `plot_1d_bo` (supports `y_scale`)
- **2D**: `plot_2d_bo` (supports `y_scale`)
- **3D+**: `plot_2d_bo`/`plot_1d_bo` don't apply. Use `plot_nd_slices`
  (supports `y_scale`) — one 1D slice per dimension through the current
  best point, holding the rest fixed. Pass `dims=[...]` with the most
  sensitive dimensions (from `get_length_scales`) if D is large enough that
  plotting every dimension is unwieldy.

## Known gaps / things not yet built

- No `y_scale` support in the iteration-history plots (see above). Only
  affects the functions that need scaling at all; elsewhere the axis units
  are already the real ones.
- ~~No persistence of `x_next` between sessions~~ — **done**, via the
  `weekly_data/` CSV store described above.
- The acquisition backtest simulates one-step lookahead only — it hasn't
  been extended to simulate multi-step BO runs using the GP's own posterior
  as a stand-in oracle (discussed as a possible future step, not built).
