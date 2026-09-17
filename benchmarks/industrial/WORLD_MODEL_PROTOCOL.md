# Experimental state-space world-model protocol

Written before fitting or held-out measurement, 2026-09-17. This is an adapter
and measurement experiment on an established environment, not a new world-model
benchmark or a claim of general robotics capability.

## Reuse and task

Use Gymnasium's actual `CartPole-v1` transition implementation and lifecycle;
do not reproduce its physics as the evaluation oracle. Fit four independent
scikit-learn `BayesianRidge` regressors to one-step state deltas. Record upstream
versions, environment source hash, all transitions and fitted coefficients.
The state coordinates are cart position (m), cart velocity (m/s), pole angle
(rad) and pole angular velocity (rad/s). Actions map 0/1 to -1/+1 for regression.

Control models: the same regression procedure without an action feature, and
state persistence (every future equals the initial state). Do not equate an
agent reasoning trace evaluator with a dynamics model evaluator.

Native references:
- https://gymnasium.farama.org/environments/classic_control/cart_pole/
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html

## Frozen partitions and measurements

- Training: reset seeds 1000–1199, uniform random actions, at most 200 steps per
  episode. A seeded NumPy generator with seed `reset_seed + 100000` supplies
  action sequences. Stop at actual termination/truncation.
- Development smoke check: reset seeds 2000–2004. Use these only to debug the
  harness; record changes before opening held-out results.
- Held-out open-loop predictions: reset seeds 3000–3039, same action-generation
  rule, at most 32 observed transitions. Measure horizons 1, 4, 8, 16, 32. Report
  denominator and censored counts at each horizon; do not step past termination
  or substitute an invented absorbing state. Test episodes are new seeds in the
  same environment, not new physics, topology or task families.
- Action interventions: each held-out reset seed, actual single-step rollouts
  for actions 0 and 1 from the matched reset state. Compare predicted effect
  vectors with simulator effect vectors; retain model and reference values.
- State-offset persistence: each held-out seed, initial cart-position offset
  +0.5 m, same recorded action prefix. Use an explicitly named reset wrapper
  around the actual environment. Compare the predicted and observed change
  caused by the offset. This is a fully observed state diagnostic, not hidden
  object memory or partial-observability evaluation.
- Uncertainty: report nominal 90% marginal Gaussian interval coverage, width
  and marginal negative log density separately by coordinate and horizon.
  Bayesian one-step variances propagate through a linear covariance recursion
  with independent innovations. That approximation ignores temporal parameter
  correlations; coverage is an empirical diagnostic, not a guarantee.
  Persistence supplies no uncertainty; missing uncertainty remains unavailable.
- Planning: seeds 4000–4009, three model conditions, fresh environments, at
  most 200 real steps each. Replan every step over all 64 binary action sequences
  of length 6 using SciPy `brute(..., finish=None)`. Mean predicted state cost
  uses weights `[1, 0.05, 20, 0.5]` and action penalty 0.01. Identical action
  costs use upstream deterministic tie handling. Record every executed action,
  observed state, reward, termination, truncation and planner failure. Report
  bounded steps survived/return, per-seed paired differences and count reaching
  the cap. Reaching 200 steps is not solving CartPole's 500-step specification.

## Evidence and limits

Freeze runtime sources, this protocol and configuration before collection.
Persist training data and model parameters before held-out evaluation. Model
callbacks receive only initial observed state and actions, never future truth.
Preserve simulator errors, model errors and censored horizons separately.
Exercise deliberate error controls outside the quality sample. Retain raw
predictions, interval parameters and native episode evidence for offline replay.

Group action/offset variants by reset seed; do not count them as independent
sources. Numerical summaries carry their measured n. Do not pool unlike physical
units into an unexplained score. No tuning on held-out results, API calls,
general-model rankings, production-safety claims or independently reviewed
customer validation. Runtime changes after data collection must be disclosed.
If this setup exposes a weak model or uncalibrated intervals, publish that result.

Related-work guardrail: action-conditioned prediction and planning utility are
established research concerns, including the decision-focused position paper
https://arxiv.org/abs/2606.15032. Video benchmarks such as WorldModelBench
https://arxiv.org/abs/2502.20694 address a different output profile. Existing
WorldFoundry and WorldBench video tooling were reviewed; this bounded state-space
experiment neither replaces them nor inherits their results.
