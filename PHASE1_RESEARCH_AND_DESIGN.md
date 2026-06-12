# Phase 1 - Research & Design Checkpoint
## Recipe Prediction & Inverse Formulation Design (plant-based cheese)

Status: **PAUSED for your review.** Environment is set up, data is understood, the
state of the art is screened, and the build plan is below. Nothing has been
modelled yet - I stop here as you asked.

---

## 1. What the data actually is (schema only - no sensitive values read)

I inspected only metadata/coverage of `Analysis_all.xlsx` (counts, column names,
ingredient names, units, how many trials have each measurement). I did **not**
read or display any individual recipe composition or measurement value. All
modelling in the notebook will load it directly; I never surface its contents.

Four sheets, keyed by `Trial`:

| Sheet | Rows | Unique trials | What it holds |
|---|---|---|---|
| Formulation Data | 11,898 | 1,204 | Ingredient %s (long format). 172 ingredients, 13 categories. **Percentages sum to 1.0 per trial.** ~10 ingredients/recipe (range 1-17). |
| Physical Data | 54,275 | 702 | Measured properties (long format, multiple replicates per parameter). |
| Process Data | 4,151 | 1,120 | Per-step Temperature, Duration, Speed, Speed_Agitator (1-10 steps/trial). |
| Nutritional data | 1,195 | 1,195 | Computed nutrition + Nutri-Score (deterministic from formulation - not a prediction target). |

**Overlap that matters for modelling:** 620 trials have *both* formulation and
physical data; 609 have all three. So the usable supervised training set is
~600-620 recipes - small, exactly as you remembered.

### Measurable properties (candidate prediction targets) and their coverage

| Analysis | Parameters | ~Trials w/ data |
|---|---|---|
| Texturometer | hardness, chewiness, cohesion, gumminess, resilience, springiness, adhesiveness | ~621 |
| Oven | melting, oiling | ~610 |
| Rheometer | complex/avg viscosity, viscosity min/max, extensibility (avg/max), glass transition T, melted adhesiveness, melting speed | ~380 |
| Moisture and fat | fat, moisture | ~245 |
| Mastersizer | particle size D[3,2], D[4,3], Dx10/50/90, span, SSA, uniformity | ~9 (too few - excluded) |

Texture measurements carry a `Texture_Analysis_Type` (BLOCK ~552 trials, NONE,
SHREDS 43, SLICES 26). BLOCK dominates; mixing measurement conditions is one
source of the noise you mentioned, so I will treat measurement type explicitly
(prefer BLOCK; condition kept as context) rather than averaging across them.

### Ingredient sparsity (confirms your insight)

13 categories. Of 172 ingredients, only **82 appear in >=5 recipes**; the rest are
one/two-off test ingredients that add noise without predictive power - exactly
the problem you hit last time. Category mass is well populated though: Water
(1168), Acid (1055), Salt (1048), Oil (1129), Starch (1119), Colour (802),
Flavour (853), Protein (521), Fibre (593).

---

## 2. State of the art (web screen)

The task - *given target properties, produce the recipe that achieves them* - is
the textbook **inverse design / inverse formulation** problem. Consensus from the
recent literature:

**Two-stage "surrogate + optimiser" is the dominant, proven pattern.**
1. Train a **forward (surrogate) model**: formulation (+ process) -> properties.
2. **Invert** it with an optimiser that searches the composition space for the
   recipe whose *predicted* properties best match your targets, subject to
   mixture constraints.

- **Inverse design in food - comprehensive review** (Trends in Food Science &
  Tech, 2023): blends representation learning with constraint-aware generation;
  inputs = ingredient/compositional vectors; outputs = candidate formulations to
  a target. Confirms the surrogate+search framing as the field standard.
  https://www.sciencedirect.com/science/article/pii/S0924224423001693
- **Predicting food taste with bound-driven optimization** (arXiv 2606.20206,
  2026): *almost our exact problem* - 5 target dimensions, **only 70 recipes**,
  forward model + **Differential Evolution** inverse design with compositional
  constraints (mass fractions sum to 1, per-ingredient min/max), objective =
  weighted relative error across the 5 targets. Reports LOO MAE/PCC per
  dimension. This is the template I will adapt. DE settings they used: popsize
  15, CR 0.8, F in [0.5,1.0], best1bin, 500 iters.
  https://arxiv.org/html/2604.20206v1
- **AI for food** (npj Science of Food, 2025): ANN takes weighted ingredient
  vectors -> property vectors; the *same* trained network is inverted to find
  formulations with desired properties. Flags that formulation->texture/rheology
  data are rare (small-data regime is normal here).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12098880/
- **Tabular ML benchmarks (2024-2025):** on small tabular data, **gradient-boosted
  trees (XGBoost/LightGBM/CatBoost) and Random Forest remain the strongest, most
  reliable baselines**; deep nets rarely beat them below a few thousand rows.
  **TabPFN v2** (a tabular foundation model) is genuinely strong for <1,000 rows /
  <100 features on CPU - our regime - and is worth trying as a challenger.
  https://arxiv.org/html/2408.14817v1  |  https://github.com/PriorLabs/TabPFN
- **Surrogate-assisted Differential Evolution** for expensive constrained
  mixed-variable problems is an established inverse-design optimiser family -
  supports our DE-over-a-learned-surrogate choice.
  https://www.sciencedirect.com/science/article/abs/pii/S0020025522014906

**Takeaway:** I am not inventing an approach - I am implementing the field-standard
surrogate+DE inverse-design pipeline, tuned for small/noisy formulation data, and
benchmarking the forward model across the strongest tabular learners.

---

## 3. Proposed design (informed by your insights)

### 3.1 Your insights, and exactly how each is handled

1. **"Some ingredients used only once/twice -> weak predictive power."**
   -> **Hybrid featurisation.** Ingredients appearing in >= N recipes (default
   N=8, tunable) become individual % features. Every rarer ingredient is folded
   into its **category-mass feature** (sum of % per category) so its mass still
   counts, but its noisy individual identity does not. Result: ~50-60 individual
   ingredient features + 13 category-mass features instead of 172 sparse columns.

2. **"Grouping by category might help, but some starches are very different."**
   -> The hybrid scheme is the answer: common, distinct starches keep their own
   columns; only obscure one-offs collapse to "Starch %". You get category
   structure *without* erasing the differences that matter. I'll also report
   accuracy with category-only vs hybrid vs full features so we can *see* whether
   grouping helped, rather than assume.

3. **"Data is noisy - same recipe, different measurements (human/machine)."**
   -> Three defences:
   (a) **Quantify the noise floor first.** From replicate measurements I compute
   per-parameter repeatability -> an estimated **noise ceiling** (the best R-squared any
   model could reach). "Predictability" is judged against this ceiling, not 1.0.
   (b) **Aggregate replicates** (robust mean/median) per trial+parameter+condition.
   (c) **Grouped cross-validation on a recipe fingerprint** so identical/near-
   identical formulations can't sit in both train and test folds (which would
   fake high scores). This is likely *why prior attempts looked off*.

### 3.2 Pipeline

```
Analysis_all.xlsx
   |
   |-- Formulation -> wide hybrid feature matrix (individual + category mass)
   |-- Process     -> per-trial numeric summaries (see 3.3)
   |-- Physical    -> replicate-aggregated targets (wide), + noise ceiling
   v
[A] FORWARD MODEL (per property)
     compare: Ridge/ElasticNet, RandomForest, ExtraTrees, HistGBR,
              XGBoost, LightGBM, (+ TabPFN challenger)
     eval: repeated GroupKFold CV (grouped by recipe fingerprint)
     metrics: R-squared, RMSE, MAE  vs  noise ceiling
   |
   v
[B] PREDICTABILITY RANKING -> auto-select TOP 5 properties (overridable cell)
   |
   v
[C] INVERSE DESIGN  (target props -> optimised recipe)
     warm start: nearest real recipes in target-property space
     optimiser: Differential Evolution over a realistic ingredient palette
     constraints: % >= 0, sum = 100%, per-ingredient bounds from data
     co-optimise standardised process params (temp / time / speed)
     objective: weighted relative error across the 5 targets (surrogate-scored)
   |
   v
   Optimised recipe (ingredients % + process settings)
   + predicted property profile vs your targets + nearest real recipe for sanity
```

### 3.3 Process handling (per your steer)

Process *steps* aren't standardised, but the numeric parameters are. I'll derive
per-trial summaries - **max temperature, total duration, max mixing speed, max
agitator speed, number of steps** - use them as forward-model features, and
co-optimise the continuous ones in the inverse step. Step semantics are ignored
as features but the closest real recipe's step sequence is shown alongside the
output so the result stays actionable.

### 3.4 Deliverable

One self-contained notebook, `Recipe_Prediction.ipynb`, runnable on the `.venv`,
with: data prep -> noise analysis -> forward-model benchmark -> predictability
ranking -> interpretability (SHAP/importances) -> inverse-design engine -> an
interactive "set your 5 targets, get a recipe" cell -> saved artifacts
(models, metrics, ranking). Honest accuracy reporting against the noise ceiling.

### 3.5 Honest expectations

With ~600 noisy recipes, expect some properties to predict well (texture
hardness, melting, moisture/fat are likely candidates) and others to sit near
their noise floor. The notebook will tell us *which* are trustworthy - that
ranking is itself a key result and drives the choice of the 5 design inputs.

---

## 4. Open questions for you (optional - I have sensible defaults for all)

1. **Target priority:** any properties you specifically care about designing for
   (e.g. melt + hardness)? Default = auto-select the 5 most predictable.
2. **Min-frequency N** for an ingredient to get its own feature: default 8.
   Lower = more detail/noise; higher = more grouping.
3. **Multiple recipes per target?** I can return the single best recipe or a few
   diverse candidates. Default = best + 2 alternatives.

I'll proceed with the defaults if you just say "go".

---

## 5. Environment (done)

- `.venv` created (Python 3.12) with pandas, numpy, scikit-learn, xgboost,
  lightgbm, scipy, optuna, shap, jupyter. Full pin list in `requirements.txt`.
- TabPFN is **not** installed yet (optional challenger; pulls in torch). I'll add
  it behind a try/except so the notebook runs with or without it.
