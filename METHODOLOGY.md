# Methodology — Recipe Prediction & Inverse Formulation Design

Technical companion to `Recipe_Prediction.ipynb`. Every major decision in the
pipeline, what was chosen, what was rejected, and why. Literature grounding is
in `research.md` (40 cited sources); this file explains how those findings were
applied to *this* dataset.

---

## 1. Problem framing: two-stage surrogate + optimiser

**Task.** Given ~1,200 plant-based-cheese trials (ingredient percentages,
process parameters, measured physical properties), (a) predict properties from
formulation, (b) invert: given target property values, output an optimised,
physically credible recipe.

**Architecture: a *forward surrogate* (recipe -> properties) wrapped by an
*inverse optimiser* (properties -> recipe).** This is the field-standard
pattern for small-data formulation design (research.md S5). Alternatives
considered and rejected:

- **Direct inverse regression** (properties -> recipe as a supervised model):
  ill-posed — many recipes give the same property profile, so the conditional
  mean of recipes is itself usually not a valid recipe; and constraints
  (sum-to-100%, category limits) cannot be guaranteed by a regressor.
- **Generative models** (cVAE / GAN / diffusion conditioned on properties):
  need orders of magnitude more data than ~700 unique recipes to learn a
  usable conditional distribution, and provide no constraint guarantees.
- The surrogate+optimiser split keeps each part testable: the surrogate has a
  measurable cross-validated accuracy, and the optimiser is a deterministic,
  inspectable search over that surrogate.

## 2. Data handling decisions

| Decision | Why |
|---|---|
| Replicates aggregated by **median** per (trial, parameter) | Robust to single bad measurements; the replicate scatter is reused for the noise ceiling (S4) rather than discarded. |
| Texturometer restricted to **BLOCK** measurements | Mixing measurement geometries within one target would conflate instrument condition with formulation effects. |
| pH taken from the per-trial Nutritional sheet (859 trials) | Treated as a property like any other: benchmarked, targetable, and reused as a realism constraint (S9). |
| Sensitive source workbook never displayed | All notebook output is aggregate statistics; raw rows are never printed. Data files are gitignored. |

## 3. Featurisation (hybrid; 115 features)

Driven by two failure modes seen in a previous in-house attempt: rare
ingredients destroy predictive power, and naive category grouping erases
signal (e.g. starches differ strongly from each other).

- **Individual % columns only for ingredients used in >= 5 recipes** (82 of
  172). An ingredient seen once or twice has no statistical support; its
  column would only add variance.
- **13 category mass totals** (`cat__Starch`, `cat__Acid`, ...) — these absorb
  the rare ingredients, so a one-off starch still contributes to "starch
  load" without getting its own unlearnable column. Keeping *both* individual
  and category features lets the model use whichever level generalises.
- **Ingredient-spec features** (from the `Ingredient_info` sheet): per-recipe
  physical-state fractions (liquid/powder/solid/paste mass shares) and the
  mixed nutrient composition (fat, protein, carbs, salt... per 100 g,
  linearly mixed from supplier specs). These describe *any* recipe physically
  — including rare-ingredient recipes — and they are what the realism
  constraints (S9) operate on. Measured effect on CV R²: roughly neutral
  (+/-0.01); their value is constraint expressiveness, not accuracy.
- **Process summaries** (max/mean temperature, total duration, max speeds,
  step count). Process *steps* are free-text and non-standardised, so only
  numeric summaries are used — a deliberate information/noise trade-off.
- **Ingredient count** (`n_ingredients`) as an explicit complexity feature.

## 4. Noise ceiling: how good can any model be?

The same recipe measured twice gives different numbers (machine + human
variation). Before judging models, the explainable variance is estimated per
property from replicates, ICC-style:

    R2_ceiling = sigma2_between_trials / (sigma2_between + sigma2_within)

Examples: Moisture 0.989, hardness 0.961, viscosity max 0.950, gumminess
0.833, ... springiness 0.564. **Model R² is judged relative to this ceiling**:
the selected top-5 models capture 69–83% of their ceilings, i.e. a large share
of the *remaining* error is measurement noise, not model failure. The ceiling
also acts as a leakage alarm — a CV score *above* its ceiling would indicate
duplicate-recipe leakage.

## 5. Validation protocol (the part most pipelines get wrong)

- **1,204 trials but only 719 unique recipes** — replicated batches are exact
  duplicate formulations. Random K-fold would put copies of the same recipe
  in both train and test and inflate scores. Every CV split is therefore a
  **GroupKFold over an MD5 fingerprint of the rounded ingredient vector**, so
  a recipe never appears on both sides. Repeated 2x with reshuffling, scores
  averaged.
- **No hyperparameter tuning.** All models run fixed, conservative,
  regularised settings (depth 4, min-samples leaves, L2, subsampling). This
  forgoes a little accuracy to eliminate tuning-selection overfitting — with
  20+ properties and 6 models, per-property tuning would overfit the
  benchmark itself.
- Reported metrics: out-of-fold R², RMSE, MAE. Final models are refit on all
  data for deployment (standard); their honest accuracy estimate remains the
  CV number.

## 6. Forward model families and why these six

Chosen for small (n = 200–900 per property), noisy, tabular, mixed-scale data
— the regime where gradient-boosted trees and bagged trees are state of the
art and deep nets underperform (research.md S1):

| Family | Role |
|---|---|
| **ElasticNetCV** (L1+L2 linear) | Strong baseline; wins when the response is near-linear in composition (it won `max extensibility`). Internal CV picks regularisation per fold. |
| **RandomForest / ExtraTrees** (bagged trees) | Low-variance averaging over interactions; ExtraTrees' extra randomisation helps under noise (won hardness, gumminess, viscosity max). |
| **HistGradientBoosting / XGBoost / LightGBM** (boosting) | Bias reduction with strong regularisation; native missing-value handling (XGBoost won pH; LightGBM/RF variants won Moisture across versions). |
| **TabPFN v2** (optional, auto-detected) | Transformer prior-fitted on synthetic tabular tasks; competitive at exactly this sample size. Not installed by default; joins the benchmark if present. |

Rejected: deep MLPs (data-hungry, unstable on n~500), Gaussian processes
(O(n^3) is fine here but kernel choice over 115 mixed sparse features is
fragile; trees dominated in pilot results from the literature).

**The benchmark is per-property** — the best family is kept for each property
(there is no reason one inductive bias should win everywhere, and it doesn't).

Headline results (GroupKFold CV R², executed notebook): pH 0.849 (XGBoost,
n=859), hardness 0.744 (ExtraTrees), Moisture 0.729, max extensibility 0.719
(ElasticNet), gumminess 0.697 (ExtraTrees). Worst: adhesiveness ~0.16,
springiness ~0.22 — flagged as not usefully predictable.

## 7. Target selection

The tool auto-selects the **top-5 properties by CV R²** (currently pH,
hardness, Moisture, max extensibility, gumminess) — "give me the most
predictable parameters" — overridable via `SELECTED_PROPERTIES`. For cheese
archetypes (S10), a per-archetype rule applies: properties with R² < 0.4 are
dropped automatically and the best-predictable remaining ones are used
(at most `max_targets`, default 5). Rationale: optimising toward a model that
cannot predict (springiness at R² 0.22) just injects noise into the search.

## 8. Inverse optimisation: differential evolution, and why

**Why a population/evolutionary method at all:** the best surrogates are tree
ensembles, whose prediction surface is **piecewise-constant — gradient
methods see zero gradient almost everywhere**, so gradient/SLSQP-style
optimisation is structurally wrong here. Bayesian optimisation targets
expensive black boxes (it spends compute deciding where to evaluate); the
surrogate is cheap to evaluate in batch, which favours a population search.
`scipy.optimize.differential_evolution` (strategy `best1bin`) was chosen as a
robust, well-understood global optimiser with native bound support
(research.md S5 — DE is the most used optimiser in surrogate-assisted
formulation studies).

Key implementation decisions:

- **Vectorised evaluation** (`vectorized=True`): one objective call scores the
  whole population through batched model predictions. This is the difference
  between seconds and hours per generation (single-row tree predictions have
  ~ms overhead each; a population is hundreds of candidates × 150
  generations).
- **`polish=False`**: the default L-BFGS-B polish is useless on a
  piecewise-constant surface (zero gradients) and was the dominant runtime
  cost when enabled.
- **Sum-to-100% by normalisation repair**: decision variables are raw weights
  in [0,1], normalised to fractions inside the objective. Simpler and better-
  behaved for DE than a linear equality constraint, and it cannot be violated.
- **Top-K sparsity projection**: only the K largest ingredient weights
  survive (K = median ingredient count of the neighbour recipes + 2; the rest
  are zeroed before normalisation). Replaces a soft "too many ingredients"
  penalty that DE simply paid instead of obeying. Side benefit: keeps the
  `n_ingredients` feature inside the training distribution, where the
  surrogate is valid.
- **Warm start, three layers** (k-NN in standardised target-property space,
  k=30): (1) the neighbours' ingredients define the **palette** (search
  space restriction to combinations that plausibly co-occur); (2) their
  process settings define the **process bounds**; (3) — decisive — **the
  neighbours' actual recipes seed the DE start population** (plus jittered
  copies). Real recipes satisfy every realism band by construction, so the
  search starts feasible and only refines toward targets. Measured effect on
  the mozzarella archetype: objective 8.36 -> 1.45 with all constraints
  satisfied.
- **Multi-restart + dedupe**: independent DE runs from differently jittered
  populations; results sorted by objective and deduplicated by ingredient
  set; the best plus alternatives are returned (different local optima =
  genuinely different formulation strategies).
- **Objective scaling**: each property's error is divided by its **IQR across
  all trials**, not by the target value. Relative error explodes when a
  target sits near the low end of its range (mozzarella viscosity made the
  objective meaningless at ~50); IQR scaling makes errors comparable across
  properties and robust at distribution edges. Per-property user weights
  multiply on top.

## 9. Realism constraints (the surrogate will be exploited otherwise)

An optimiser pointed at an imperfect surrogate finds its blind spots — it
will propose recipes that score well *in the model* but are physically
absurd. Each constraint below was added after observing exactly that failure:

| Constraint | Failure it fixes | Form |
|---|---|---|
| **Liquid-fraction band** (state specs; neighbours' min–max) | A "powder brick" candidate with ~16% liquid that could never mix (real recipes: 37–75% liquid, median 51%) | Penalty 20x outside band |
| **Per-category caps** (neighbours' max, clipped by global p99) | A 10% acid dose (real recipes: median 0.13% acid, p99 1.19%); the clip stops extreme one-off test trials from loosening the cap | Penalty 200x above cap |
| **pH band** (dedicated LightGBM pH model; neighbours' measured pH range clipped to global p5–p95) | Chemically implausible acid/pH combinations; gives the optimiser "pH understanding" | Penalty 20x outside band |
| **Per-ingredient cap** (1.5x its max observed share) | Single-ingredient extrapolation beyond anything ever trialled | Penalty 5x on excess |
| **Top-K ingredient cap** (structural) | 29-ingredient "smear" recipes; also keeps `n_ingredients` in-distribution | Hard projection |

Penalty weights are deliberately asymmetric: category caps (200x) are
near-hard constraints (the surrogate had learned acid->stretch correlations
and happily *paid* a 50x penalty to exploit them), while bands that interact
with targets (pH, liquid) are strong-but-tradeable. Upper-only category caps:
two-sided bands conflicted with the top-K cap (neighbours always carry trace
flavour/colour, which a lean candidate cannot all include).

Every candidate is reported with its liquid/powder/solid/paste split,
estimated pH, and acid share, so a human can sanity-check at a glance.

## 10. Cheese archetypes ("latent space" targeting)

The property table is used as an interpretable latent space: a point in it is
a quantitative "idea of a cheese". 258 commercial dairy cheeses (8 types —
CHEDDAR 97, GOUDA 40, MOZZA 28, PARMESAN 27, HALLOUMI 22, FETA 22, BURGER
SLICES 15, TILSITER 7), measured with the **same instruments and protocol**
as the trials (verified: all 18 columns map 1:1 to trial parameters and value
ranges nest inside trial ranges), define type-level **median profiles**.
`design_for("MOZZA")` = inverse-design toward that profile.

- **No training contamination**: the cheese trials have no formulation rows,
  so they cannot enter the supervised set (verified 0/258) — they are pure
  reference targets.
- **Why medians** per type: robust to outlier products; a type is a cloud,
  its median is the archetype.
- **Predictability filter** per archetype (S7) keeps the optimisation
  well-posed.
- **Residuals are information**: the mozzarella design reaches the hardness
  target but only ~half the stretch target — meaning the explored
  ingredient/process space does not yet contain full mozzarella stretch.
  Archetype residuals literally map which cheese "ideas" are reachable from
  the current formulation space and which need new trials.

A deeper learned latent space (autoencoder over recipes) was considered and
deferred: at ~700 unique recipes it would add little over the explicit,
interpretable property space, which optimises and explains better.

## 11. Known limitations (say these to the expert before they ask)

1. **Correlational, not causal.** The surrogate learns co-occurrence in
   historical trials; constraints keep it honest, but a designed recipe is a
   *hypothesis to test*, not a guarantee.
2. **No extrapolation.** Targets outside observed property ranges (parmesan
   hardness) yield best-effort approximations; the predicted-vs-target gap
   quantifies the shortfall.
3. **Mixability is a mass balance, not kinetics.** The liquid band cannot see
   mixing order, hydration speed, or shear history.
4. **Process is summarised**, not sequenced: step-order effects are invisible
   to the model.
5. **Single final fit.** Final models are refit on all data; the honest
   accuracy estimate is the GroupKFold CV number. A stricter audit would be a
   temporal holdout (train on older trials, test on the newest).
6. **The natural next step is a closed loop**: design -> make -> measure ->
   append to the dataset -> retrain. Each verified candidate is the most
   informative possible new training point near the targets (this is
   active learning / sequential model-based optimisation in spirit).

## 12. Reproducibility

- Python 3.12 venv (`requirements.txt`), fixed seeds (`RNG = 42`) for models,
  CV shuffling, and DE.
- Full notebook execution: ~60 min (benchmark ~35 min; two optimisation
  demos ~15 min). Models + metadata exported to `models/` (joblib).
- Data files (`*.xlsx`) are intentionally untracked in git; place them in the
  repo root before running.

## Glossary (one-liners)

- **Surrogate model** — fast learned stand-in for the real process (recipe -> properties).
- **R²** — share of variance explained; 1 is perfect, 0 is "predicts the mean".
- **ICC / noise ceiling** — max R² achievable given measurement repeatability.
- **GroupKFold** — cross-validation that keeps all copies of a group (here: a recipe) in the same fold.
- **Differential evolution** — population-based global optimiser; mutates and recombines candidates, keeps improvements; needs no gradients.
- **IQR** — interquartile range (p75 - p25), a robust measure of a property's spread.
