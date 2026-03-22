# Testing

## Running tests

```bash
python3 -m pytest backend/tests/ -v
```

To run a single test file:

```bash
python3 -m pytest backend/tests/test_avoid_ingredients.py -v
python3 -m pytest backend/tests/test_nutrient_targets.py -v
python3 -m pytest backend/tests/test_recommender.py -v
```

To run a single test class or method:

```bash
python3 -m pytest backend/tests/test_nutrient_targets.py::TestHypertension -v
python3 -m pytest backend/tests/test_recommender.py::TestSafeRecipesFiltering::test_hypertension_removes_salty_recipes -v
```

### Prerequisites

```bash
pip install pytest pytest-asyncio
```

No database or running server is required — all tests run against pure Python functions or use mocks.

---

## Test files

### `test_avoid_ingredients.py`

Tests `get_avoid_ingredients()` from `avoid_ingredients.py`, which maps health conditions to ingredient names that should be excluded from recipes.

| Test class | What it covers |
|---|---|
| `TestGetAvoidIngredients` | 19 tests |

**Key tests:**

- **Return schema** — verifies the function returns a dict with `unified` (set) and `by_condition` (dict) keys
- **Empty inputs** — empty conditions returns empty results; empty ingredient list returns empty results
- **Hypertension** — flags `Salt` and `Soy sauce`; does not flag `Salmon fillet`
- **Diabetes (Type 2)** — flags `Corn syrup` and `White sugar`; does not flag `Spinach`
- **Heart Disease** — flags `Butter`
- **Thyroid Disorder** — flags `Kelp` and `Tofu`
- **Multi-condition union** — combining Hypertension + Diabetes flags ingredients from both conditions
- **Per-condition breakdown** — `by_condition` dict has a key for each input condition, and each list is sorted
- **Unified superset** — `unified` is always a superset of all per-condition lists
- **Exclude patterns** — `Peanut butter` is not flagged by Heart Disease's butter rule (exclude pattern prevents the false positive)
- **Unknown condition** — unrecognized condition names return empty results without errors

---

### `test_nutrient_targets.py`

Tests `compute_nutrient_targets()` from `nutrient_targets.py`, which calculates daily nutrient intake targets from biometric data and health conditions using Mifflin-St Jeor BMR and DRI-based adjustments.

| Test class | What it covers |
|---|---|
| `TestComputeNutrientTargetsBaseline` | 8 tests — no health conditions |
| `TestSexDifferences` | 3 tests — male vs female |
| `TestActivityLevels` | 2 tests — sedentary through very active |
| `TestHypertension` | 2 tests |
| `TestHeartDisease` | 2 tests |
| `TestDiabetes` | 2 tests |
| `TestKidneyDisease` | 3 tests |
| `TestWeightManagement` | 2 tests |
| `TestConflictResolution` | 3 tests — multiple conditions |

**Baseline tests:**

- Returns all 9 expected keys (`kcal`, `protein_g`, `carbs_g`, `fat_g`, `saturated_fat_g`, `fiber_g`, `sugar_g`, `sodium_mg`, `cholesterol_mg`)
- All values are positive numbers
- `kcal` is within a reasonable range (1200–4000)
- Baseline sodium is 2300 mg (DGA upper limit)
- Baseline cholesterol is 300 mg (NCEP ATP III)
- Carbs are at least 130g (IOM RDA minimum for brain glucose)

**Sex and activity:**

- Males get higher kcal, fiber, and sugar caps than females
- Higher activity levels produce higher caloric targets

**Condition-specific adjustments:**

- **Hypertension** — sodium lowered to 1500 mg; saturated fat reduced
- **Heart Disease** — cholesterol lowered to 200 mg; sodium lowered to 1500 mg
- **Diabetes** — sugar lowered to 25g; carbs reduced
- **Kidney Disease** — protein capped below 1.0 g/kg; sodium lowered to 2000 mg; kidney cap overrides Weight Management's protein increase
- **Weight Management** — kcal reduced by 500; protein increased

**Conflict resolution:**

- Sodium takes the lowest value across conditions (Hypertension 1500 beats Kidney Disease 2000)
- Fiber takes the highest value (more fiber is universally beneficial)
- Pregnancy calorie surplus (+350) overrides Weight Management deficit (-500)

---

### `test_recommender.py`

Tests the recommendation pipeline functions from `backend/app/recommender.py`: recipe filtering, ingredient-nutrient mapping, nutrient-distance scoring, user target computation, and KG-based recommendations.

| Test class | What it covers |
|---|---|
| `TestSafeRecipesFiltering` | 8 tests — avoid_ingredients recipe filtering |
| `TestRecipeIngredientNutrients` | 7 tests — ingredient-to-nutrient mapping |
| `TestNutrientRelevanceScore` | 8 tests — nutrient-distance scoring |
| `TestComputeUserTargets` | 7 tests — biometric target computation |
| `TestKGRecommendations` | 3 tests — mocked KG engine |

**Safe recipes filtering** — replicates the filtering pattern from `get_recommendations()`:

- No conditions returns all recipes unchanged
- Hypertension removes recipes containing salt/soy sauce; keeps salmon
- Diabetes removes recipes containing sugar/corn syrup; keeps lentil soup
- Multiple conditions filter more aggressively than single conditions
- Recipes with no ingredients survive filtering
- Empty recipe list returns empty

**Ingredient-nutrient mapping** — tests `_recipe_ingredient_nutrients()`:

- Salmon maps to `protein` and `fat`
- Lentils map to `fiber` and `iron`
- Salt maps to `sodium`; sugar maps to `sugar`
- Empty ingredient list returns empty set

**Nutrient-distance scoring** — tests `_nutrient_relevance_score()`:

- Recipes providing needed nutrients (protein, fiber, iron) score lower (better) than empty recipes
- Recipes with only capped nutrients (salt) score higher (worse) than empty recipes
- More restrictive sodium caps produce larger penalties
- Default caps (2300 mg sodium) produce no penalty
- Empty recipes score exactly 0.0
- Nutrient-rich recipes (lentil, salmon) rank better than empty recipes

**User target computation** — tests `_compute_user_targets()` wrapper:

- Returns a target dict when all biometric fields are present
- Returns `None` when any field (height, weight, age, sex, activity) is missing
- Passes health conditions through to `compute_nutrient_targets` (e.g., hypertension lowers sodium)

**KG recommendations** — tests `_kg_recommendations()` with a mocked KG engine and mocked TheMealDB:

- Returns recipes sorted by KG score (lowest/best first)
- Skipped recipe IDs are excluded from results
- Previously cooked recipes are flagged with `previously_cooked: true`
- `restrictions_applied` is passed through to each recipe
