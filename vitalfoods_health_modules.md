# VitalFoods Health Modules
## `nutrient_targets.py` · `avoid_ingredients.py`

---

## Overview

Two cooperating modules translate a user's biometric and health profile into structured dietary constraints that gate the VitalFoods recommendation pipeline at two stages:

1. **`get_avoid_ingredients`** — produces a set of clinically contraindicated ingredient names matched directly against the recipe dataset. This is applied **first**, before the RotatE knowledge graph model generates candidates, so the embedding model only ever scores recipes that are safe for the user's conditions.

2. **`compute_nutrient_targets`** — produces a nine-element numeric target vector derived from validated metabolic formulas and Dietary Reference Intakes (DRIs). This drives the **final ranking step**, scoring RotatE-generated safe candidates by weighted distance to the user's personalised nutritional targets.

```
User Profile
    │
    ├─► get_avoid_ingredients() ──► ingredient block set ──► filter recipe pool
    │                                                               │
    │                                                     RotatE embedding model
    │                                                               │
    └─► compute_nutrient_targets() ──► target nutrient vector ──► weighted-distance ranking
                                                                    │
                                                            Top-N recommendations
```

---

## Module 1 — `avoid_ingredients.py`

### Function signature

```python
def get_avoid_ingredients(
    health_conditions: set[str],
    ingredient_list: Iterable[str],
) -> dict
```

### Return schema

| Key | Type | Description |
|---|---|---|
| `unified` | `set[str]` | Union of all condition avoid lists — used to filter the recipe pool before RotatE |
| `by_condition` | `dict[str, list[str]]` | Per-condition breakdown for explainability and UI display |

### Design

Each health condition owns one or more `ConditionRule` instances. A rule declares:

- **`include_patterns`** — regex patterns; an ingredient is a candidate to avoid if *any* fires.
- **`exclude_patterns`** — regex patterns; a candidate is kept in the diet if *any* fires.

All matching runs case-insensitively on the normalised ingredient name. Multiple rules per condition are unioned into the condition's avoid set. Exclude patterns are essential for precision — they prevent false positives caused by substring collisions in real ingredient names (e.g. `"peanut butter"` matching `butter`, `"ginger ale"` matching `\bale\b`, `"reduced-sodium soy sauce"` matching `soy sauce`, `"lemon juice"` matching the fruit-juice rule, `"avocado oil"` matching `avocado`).

### Conditions covered and their clinical basis

| Condition | Dietary guidance source |
|---|---|
| **Hypertension** | DASH trial (Sacks et al., *NEJM* 2001); NHLBI DASH Eating Plan; Appel et al., *Hypertension* 2006; 2025 AHA/ACC Hypertension Guidelines |
| **Heart Disease** | AHA 2021 Dietary Guidance (Lichtenstein et al., *Circulation* 2021); NCEP ATP III TLC Diet; AHA 2017 Presidential Advisory (Sacks et al., *Circulation* 2017); Carson et al., *Circulation* 2020 |
| **Diabetes (Type 2)** | ADA Standards of Care 2024–2026, Section 5; ADA/EASD Nutrition Consensus (Evert et al., *Diabetes Care* 2019) |
| **Kidney Disease** | KDOQI 2020 (Ikizler et al., *AJKD* 2020); KDIGO 2024 |
| **Weight Management** | NHLBI Obesity Guidelines; Morton et al., *Br J Sports Med* 2018 |
| **Osteoporosis** | Bone Health & Osteoporosis Foundation; IOM 2011 DRIs for Ca & D |
| **Anemia** | NIH ODS Iron Fact Sheet; WHO Iron Deficiency Anaemia Guidelines |
| **Thyroid Disorder** | American Thyroid Association; Endocrine Society (Jonklaas et al., *Thyroid* 2014) |
| **Immune Deficiency** | NIH ODS Fact Sheets; Gombart et al., *Nutrients* 2020 |
| **Inflammation** | Koelman et al., *Adv Nutr* 2022; AHA 2021; Calder, *AJCN* 2006; Simopoulos, *Biomed Pharmacother* 2002 |
| **Cognitive Decline** | MIND diet (Morris et al., *Alzheimers Dement* 2015); van den Brink et al., *Adv Nutr* 2019 |
| **Depression** | SMILES Trial (Jacka et al., *BMC Med* 2017); WFSBP/CANMAT 2022; Liao et al., *Transl Psychiatry* 2019 |
| **Fatigue** | Tardy et al., *Nutrients* 2020 |
| **Cramps** | Nosaka et al., *J Int Soc Sports Nutr* 2021; NIH ODS Mg, K, Ca Fact Sheets |
| **Muscle Weakness** | ESPEN (Bauer et al., *JAMDA* 2013); EWGSOP2 2019 |
| **Wound Healing** | NPUAP/EPUAP/PPPIA 2019; ASPEN (McClave et al., *JPEN* 2016) |
| **Hair Loss** | Almohanna et al., *Dermatol Ther* 2019; Rushton, *Clin Exp Dermatol* 2002; AAD |
| **Skin Conditions** | Dall'Oglio et al., *Int J Dermatol* 2021; Huang et al., *Mar Drugs* 2018 |
| **Pregnancy Nutrition** | ACOG Committee Opinion No. 804; IOM 2006; FDA/EPA 2024 fish advisory |

### Usage

```python
from avoid_ingredients import get_avoid_ingredients

result = get_avoid_ingredients(
    health_conditions={"Hypertension", "Diabetes (Type 2)"},
    ingredient_list=unique_ingredients,   # list[str] from dataset
)

# Filter the recipe pool before passing to RotatE
avoid_set = {x.lower() for x in result["unified"]}
safe_recipes = [
    r for r in all_recipes
    if not any(ing.lower() in avoid_set for ing in r.ingredients)
]

# Per-condition breakdown for UI explainability
for condition, flagged in result["by_condition"].items():
    print(f"{condition}: {len(flagged)} ingredients to avoid")
```

---

## Module 2 — `nutrient_targets.py`

### Function signature

```python
def compute_nutrient_targets(
    height_cm: float,
    weight_kg: float,
    age: int,
    sex: str,                  # "male" | "female"
    activity_level: str,       # "sedentary" | "light" | "moderate" | "active" | "very_active"
    health_conditions: set[str],
) -> dict
```

### Return schema

| Key | Unit | Description |
|---|---|---|
| `kcal` | kcal/day | Total daily energy target |
| `protein_g` | g/day | Protein target |
| `carbs_g` | g/day | Total carbohydrate target |
| `fat_g` | g/day | Total fat target |
| `saturated_fat_g` | g/day | Saturated fat cap |
| `fiber_g` | g/day | Dietary fibre target |
| `sugar_g` | g/day | Added-sugar cap |
| `sodium_mg` | mg/day | Sodium cap |
| `cholesterol_mg` | mg/day | Dietary cholesterol cap |

### Step 1 — Basal Metabolic Rate (Mifflin-St Jeor)

BMR is computed with the **Mifflin-St Jeor equation** (Mifflin et al., 1990), validated by Frankenfield et al. (2005) as the most accurate predictive equation for resting metabolic rate and designated the evidence-based standard by the Academy of Nutrition and Dietetics.

$$\text{BMR}_{\text{male}} = 10 \times \text{wt} + 6.25 \times \text{ht} - 5 \times \text{age} + 5$$

$$\text{BMR}_{\text{female}} = 10 \times \text{wt} + 6.25 \times \text{ht} - 5 \times \text{age} - 161$$

### Step 2 — Total Daily Energy Expenditure (TDEE)

TDEE = BMR × PAL. Physical Activity Level multipliers follow the FAO/WHO/UNU (2004) energy-requirements framework, operationalised as the five-tier clinical scale (ISSA, NASM):

| `activity_level` | PAL | Description |
|---|---|---|
| `sedentary` | 1.200 | Little or no exercise, desk job |
| `light` | 1.375 | Light exercise 1–3 days/week |
| `moderate` | 1.550 | Moderate exercise 3–5 days/week |
| `active` | 1.725 | Hard exercise 6–7 days/week |
| `very_active` | 1.900 | Very hard exercise or physical job |

### Step 3 — Baseline macronutrient targets

Baselines follow the **IOM Acceptable Macronutrient Distribution Ranges** (AMDRs) from the 2002/2005 DRI reports. Default split: 50% carbohydrate / 20% protein / 30% fat.

| Nutrient | Baseline | Source |
|---|---|---|
| Protein | 0.8 g/kg; AMDR 10–35% kcal | IOM 2005 RDA |
| Carbohydrate | 50% kcal; minimum 130 g/day | IOM 2005 AMDR & RDA |
| Fat | 30% kcal; minimum 20% kcal | IOM 2005 AMDR |
| Saturated fat | <10% kcal | 2020–2025 DGA; AHA |
| Fibre | 38 g (men ≤50), 30 g (men >50), 25 g (women ≤50), 21 g (women >50) | IOM 2005 AI (14 g/1,000 kcal) |
| Added sugar | ≤36 g men; ≤25 g women | Johnson et al., *Circulation* 2009; WHO 2015 |
| Sodium | ≤2,300 mg | NASEM 2019; 2020–2025 DGA |
| Cholesterol | <300 mg | NCEP ATP III |

### Step 4 — Condition-specific adjustments

| Condition | Numeric adjustments | Guideline |
|---|---|---|
| **Hypertension** | Sodium → 1,500 mg; sat fat → 6% kcal; fibre → 30 g | Sacks et al., *NEJM* 2001; NHLBI; 2025 AHA/ACC |
| **Heart Disease** | Sat fat → 7% kcal; cholesterol → 200 mg; sodium → 1,500 mg; fibre → 25 g | Lichtenstein et al., *Circulation* 2021; NCEP ATP III; Sacks et al., *Circulation* 2017 |
| **Diabetes (Type 2)** | Carbs → 45% kcal; sugar → 25 g; fibre → 28 g; sodium → 2,300 mg | ADA Standards of Care 2024–2026; Evert et al., *Diabetes Care* 2019 |
| **Kidney Disease** | Protein → 0.8 g/kg (**ceiling**, overrides all increases); sodium → 2,000 mg | KDIGO 2024 Rec 3.3.1.1; KDOQI 2020 |
| **Weight Management** | Calories −500 kcal; protein → 1.4 g/kg; carbs → 40% kcal; fibre → 30 g; sugar → 25 g | NHLBI Obesity Guidelines; Morton et al., *Br J Sports Med* 2018 |
| **Osteoporosis** | Sodium → 2,300 mg | BHOF; IOM 2011 |
| **Immune Deficiency** | Sugar → 25 g; protein → 1.0 g/kg | NIH ODS; Gombart et al., *Nutrients* 2020 |
| **Inflammation** | Sat fat → 8% kcal; fibre → 30 g | Koelman et al., *Adv Nutr* 2022; AHA 2021; Calder, *AJCN* 2006 |
| **Cognitive Decline** | Sat fat → 8% kcal | Morris et al., *Alzheimers Dement* 2015 |
| **Depression** | Sugar → 25 g | Jacka et al., *BMC Med* 2017; WFSBP/CANMAT 2022 |
| **Muscle Weakness** | Protein → 1.4 g/kg | Bauer et al., *JAMDA* 2013; EWGSOP2 2019 |
| **Wound Healing** | Protein → 1.5 g/kg | NPUAP/EPUAP/PPPIA 2019; ASPEN |
| **Pregnancy Nutrition** | Calories +350 kcal; protein +25 g above baseline; fibre → 28 g | ACOG; IOM 2006; NIH ODS |
| **Anemia, Thyroid Disorder, Fatigue, Cramps, Hair Loss, Skin Conditions** | No macro targets modified — constraints expressed entirely through `avoid_ingredients.py` | See condition sources in Module 1 table |

### Step 5 — Conflict resolution

When conditions request conflicting values, the function applies a deterministic, clinically conservative resolution:

| Nutrient | Rule | Rationale |
|---|---|---|
| Protein | Kidney Disease cap overrides all increases; otherwise take the highest request | Excess protein in CKD accelerates hyperfiltration (KDOQI 2020; KDIGO 2024) |
| Carbs, sat fat, sugar, sodium, cholesterol | Take the **lowest** (most restrictive) request | Most protective against the relevant clinical endpoints |
| Fibre | Take the **highest** request, floored at the IOM AI baseline | Higher fibre is universally beneficial |
| Calories | Pregnancy surplus (+350 kcal) overrides Weight Management deficit (−500 kcal) | Fetal growth takes clinical priority (ACOG) |

### Usage

```python
from nutrient_targets import compute_nutrient_targets

targets = compute_nutrient_targets(
    height_cm=175,
    weight_kg=90,
    age=55,
    sex="male",
    activity_level="light",
    health_conditions={"Hypertension", "Diabetes (Type 2)", "Weight Management"},
)

# {
#     "kcal":             1870,
#     "protein_g":        126.0,   # 1.4 g/kg — Weight Management
#     "carbs_g":          187.0,   # 40% kcal — Weight Management (lowest)
#     "fat_g":            68.7,
#     "saturated_fat_g":  12.5,    # 6% kcal — Hypertension (lowest)
#     "fiber_g":          30.0,    # Hypertension / Weight Management (highest)
#     "sugar_g":          25.0,    # Diabetes + Weight Management (lowest)
#     "sodium_mg":        1500.0,  # Hypertension (lowest)
#     "cholesterol_mg":   300.0,
# }
```

---

## Full pipeline integration

```python
from avoid_ingredients import get_avoid_ingredients
from nutrient_targets import compute_nutrient_targets

# ── Step 1: derive ingredient block set ──────────────────────────────────
avoid = get_avoid_ingredients(
    health_conditions=user.health_conditions,
    ingredient_list=UNIQUE_INGREDIENTS,
)
avoid_set = {x.lower() for x in avoid["unified"]}

# ── Step 2: filter recipe pool before RotatE ─────────────────────────────
safe_recipes = [
    r for r in all_recipes
    if not any(ing.lower() in avoid_set for ing in r.ingredients)
]

# ── Step 3: RotatE generates candidates from the safe pool ───────────────
candidates = rotate_model.recommend(
    user_entity=user.entity_id,
    recipe_pool=safe_recipes,
    top_k=200,
)

# ── Step 4: derive nutrient target vector ────────────────────────────────
targets = compute_nutrient_targets(
    height_cm=user.height_cm,
    weight_kg=user.weight_kg,
    age=user.age,
    sex=user.sex,
    activity_level=user.activity_level,
    health_conditions=user.health_conditions,
)

# ── Step 5: rank candidates by weighted distance to target vector ─────────
def nutrient_distance(recipe, targets, weights):
    return sum(
        weights[nutrient] * abs(getattr(recipe.nutrition, nutrient, 0) - target)
        for nutrient, target in targets.items()
    )

recommendations = sorted(
    candidates,
    key=lambda r: nutrient_distance(r, targets, NUTRIENT_WEIGHTS),
)[:top_n]
```

---

## Limitations

**`compute_nutrient_targets`**
- The Mifflin-St Jeor equation was validated in healthy adults aged 19–78; accuracy degrades at BMI extremes and in critically ill populations.
- The function targets macronutrients only. Micronutrient quantities central to several conditions (calcium, vitamin D, iron, zinc, omega-3 DHA/EPA, folate, B12) require separate supplementation protocols supervised by a registered dietitian.
- ADA (2024–2026) states there is no single ideal carbohydrate percentage for type 2 diabetes; the 45% figure is a practical simplification.
- Kidney Disease rules assume non-dialysis CKD stages 3–5 per KDOQI 2020. Individual management depends on GFR stage and confirmed electrolyte abnormalities.

**`get_avoid_ingredients`**
- Matching operates on ingredient name strings. Highly specific, branded, or unusually formatted names may be missed.
- Thyroid Disorder rules flag raw cruciferous vegetables; cooking substantially reduces goitrogenic activity, but preparation method is not always encoded in ingredient names.
- Kidney Disease potassium restrictions follow KDOQI 2020's liberalised guidance on plant-source potassium bioavailability. Patients with confirmed hyperkalemia may require stricter clinical management.

---

## References

### Metabolic formulas and energy requirements

1. Mifflin MD, St Jeor ST, Hill LA, Scott BJ, Daugherty SA, Koh YO. A new predictive equation for resting energy expenditure in healthy individuals. *Am J Clin Nutr.* 1990;51(2):241–247.
2. Frankenfield D, Roth-Yousey L, Compher C. Comparison of predictive equations for resting metabolic rate in healthy nonobese and obese adults. *J Am Diet Assoc.* 2005;105(5):775–789.
3. Food and Agriculture Organization / World Health Organization / United Nations University. *Human Energy Requirements: Report of a Joint FAO/WHO/UNU Expert Consultation.* FAO Food and Nutrition Technical Report Series No. 1. Rome: FAO; 2004.

### Dietary Reference Intakes (DRIs)

4. Institute of Medicine. *Dietary Reference Intakes for Energy, Carbohydrate, Fiber, Fat, Fatty Acids, Cholesterol, Protein, and Amino Acids.* Washington, DC: National Academies Press; 2005.
5. National Academies of Sciences, Engineering, and Medicine. *Dietary Reference Intakes for Sodium and Potassium.* Washington, DC: National Academies Press; 2019.
6. Institute of Medicine. *Dietary Reference Intakes for Calcium and Vitamin D.* Washington, DC: National Academies Press; 2011.

### Dietary guidelines

7. U.S. Department of Agriculture and U.S. Department of Health and Human Services. *Dietary Guidelines for Americans, 2020–2025.* 9th ed. Washington, DC; 2020.
8. World Health Organization. *Guideline: Sugars Intake for Adults and Children.* Geneva: WHO; 2015.

### Hypertension

9. Sacks FM, Svetkey LP, Vollmer WM, et al. Effects on blood pressure of reduced dietary sodium and the Dietary Approaches to Stop Hypertension (DASH) diet. *N Engl J Med.* 2001;344(1):3–10.
10. Appel LJ, Brands MW, Daniels SR, et al. Dietary approaches to prevent and treat hypertension: a scientific statement from the American Heart Association. *Hypertension.* 2006;47(2):296–308.
11. National Heart, Lung, and Blood Institute. *DASH Eating Plan.* Bethesda, MD: NHLBI; 2021.

### Cardiovascular disease

12. Lichtenstein AH, Appel LJ, Vadiveloo M, et al. 2021 Dietary guidance to improve cardiovascular health: a scientific statement from the American Heart Association. *Circulation.* 2021;144(23):e472–e487.
13. Sacks FM, Lichtenstein AH, Wu JHY, et al. Dietary fats and cardiovascular disease: a presidential advisory from the American Heart Association. *Circulation.* 2017;136(3):e1–e23.
14. Lichtenstein AH, Appel LJ, Brands M, et al. Diet and lifestyle recommendations revision 2006: a scientific statement from the American Heart Association Nutrition Committee. *Circulation.* 2006;114(1):82–96.
15. Carson JAS, Lichtenstein AH, Anderson CAM, et al. Dietary cholesterol and cardiovascular risk: a science advisory from the American Heart Association. *Circulation.* 2020;141(3):e39–e53.
16. National Cholesterol Education Program. *Third Report of the NCEP Expert Panel on Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults (Adult Treatment Panel III).* Bethesda, MD: NHLBI; 2002.

### Diabetes

17. American Diabetes Association Professional Practice Committee. Standards of care in diabetes—2024. *Diabetes Care.* 2024;47(Suppl 1):S1–S321.
18. Evert AB, Dennison M, Gardner CD, et al. Nutrition therapy for adults with diabetes or prediabetes: a consensus report. *Diabetes Care.* 2019;42(5):731–754.

### Kidney disease

19. Ikizler TA, Burrowes JD, Byham-Gray LD, et al. KDOQI clinical practice guideline for nutrition in CKD: 2020 update. *Am J Kidney Dis.* 2020;76(3 Suppl 1):S1–S107.
20. Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. KDIGO 2024 clinical practice guideline for the evaluation and management of chronic kidney disease. *Kidney Int.* 2024;105(4S):S117–S314.

### Weight management

21. National Heart, Lung, and Blood Institute. *Clinical Guidelines on the Identification, Evaluation, and Treatment of Overweight and Obesity in Adults: The Evidence Report.* Bethesda, MD: NHLBI; 1998.
22. Morton RW, Murphy KT, McKellar SR, et al. A systematic review, meta-analysis and meta-regression of the effect of protein supplementation on resistance training-induced gains in muscle mass and strength in healthy adults. *Br J Sports Med.* 2018;52(6):376–384.

### Bone health

23. Bone Health & Osteoporosis Foundation. *Clinician's Guide to Prevention and Treatment of Osteoporosis.* Washington, DC: BHOF; 2022.

### Added sugars

24. Johnson RK, Appel LJ, Brands M, et al. Dietary sugars intake and cardiovascular health: a scientific statement from the American Heart Association. *Circulation.* 2009;120(11):1011–1020.

### Immune function

25. Gombart AF, Pierre A, Maggini S. A review of micronutrients and the immune system — working in harmony to reduce the risk of infection. *Nutrients.* 2020;12(1):236.

### Inflammation

26. Koelman L, Egea Rodrigues C, Aleksandrova K. Effects of dietary patterns on biomarkers of inflammation and immune responses: a systematic review and meta-analysis of randomized controlled trials. *Adv Nutr.* 2022;13(1):101–115.
27. Calder PC. n-3 polyunsaturated fatty acids, inflammation, and inflammatory diseases. *Am J Clin Nutr.* 2006;83(6 Suppl):1505S–1519S.
28. Simopoulos AP. The importance of the ratio of omega-6/omega-3 essential fatty acids. *Biomed Pharmacother.* 2002;56(8):365–379.

### Cognitive decline

29. Morris MC, Tangney CC, Wang Y, et al. MIND diet associated with reduced incidence of Alzheimer's disease. *Alzheimers Dement.* 2015;11(9):1015–1022.
30. van den Brink AC, Brouwer-Brolsma EM, Berendsen AAM, et al. The Mediterranean, DASH, and MIND diets are associated with less cognitive decline and a lower risk of Alzheimer's disease — a review. *Adv Nutr.* 2019;10(6):1040–1065.

### Depression

31. Jacka FN, O'Neil A, Opie R, et al. A randomised controlled trial of dietary improvement for adults with major depression (the 'SMILES' trial). *BMC Med.* 2017;15(1):23.
32. Liao Y, Xie B, Zhang H, et al. Efficacy of omega-3 PUFAs in depression: a meta-analysis. *Transl Psychiatry.* 2019;9(1):190.
33. Grigolon RB, Troubat R, Brietzke E, et al. WFSBP/CANMAT guidelines for the treatment of major depressive disorder. *World J Biol Psychiatry.* 2022.

### Fatigue

34. Tardy A-L, Pouteau E, Marquez D, Yilmaz C, Scholey A. Vitamins and minerals for energy, fatigue and cognition: a narrative review of the biochemical and clinical evidence. *Nutrients.* 2020;12(1):228.

### Muscle and sarcopenia

35. Bauer J, Biolo G, Cederholm T, et al. Evidence-based recommendations for optimal dietary protein intake in older people: a position paper from the PROT-AGE Study Group. *J Am Med Dir Assoc.* 2013;14(8):542–559.
36. Cruz-Jentoft AJ, Baeyens JP, Bauer JM, et al. Sarcopenia: European consensus on definition and diagnosis (EWGSOP2). *Age Ageing.* 2019;48(1):16–31.

### Wound healing

37. European Pressure Ulcer Advisory Panel, National Pressure Injury Advisory Panel, Pan Pacific Pressure Injury Alliance. *Prevention and Treatment of Pressure Ulcers/Injuries: Clinical Practice Guideline.* 3rd ed. EPUAP/NPIAP/PPPIA; 2019.
38. McClave SA, Taylor BE, Martindale RG, et al. Guidelines for the provision and assessment of nutrition support therapy in the adult critically ill patient. *JPEN J Parenter Enteral Nutr.* 2016;40(2):159–211.

### Cramps

39. Nosaka K, Sacco P, Mawatari K. Effects of electrolyte drink on leg muscle cramps. *J Int Soc Sports Nutr.* 2021;18(1):45.

### Hair loss

40. Almohanna HM, Ahmed AA, Tsatalis JP, Tosti A. The role of vitamins and minerals in hair loss: a review. *Dermatol Ther (Heidelb).* 2019;9(1):51–70.
41. Rushton DH. Nutritional factors and hair loss. *Clin Exp Dermatol.* 2002;27(5):396–404.
42. American Academy of Dermatology. Hair loss: who gets it and causes. Available at: aad.org. Accessed March 2026.

### Skin conditions

43. Dall'Oglio F, Nasca MR, Fiorentini F, Micali G. Diet and acne: review of the evidence from 2009 to 2020. *Int J Dermatol.* 2021;60(6):672–685.
44. Huang T-H, Wang P-W, Yang S-C, Chou W-L, Fang J-Y. Cosmetic and therapeutic applications of fish oil's EPA and DHA. *Mar Drugs.* 2018;16(8):256.

### Thyroid

45. American Thyroid Association. ATA statement on the potential risks of excess iodine ingestion and exposure. Available at: thyroid.org. Accessed March 2026.
46. Jonklaas J, Bianco AC, Bauer AJ, et al. Guidelines for the treatment of hypothyroidism. *Thyroid.* 2014;24(12):1670–1751.

### Pregnancy

47. American College of Obstetricians and Gynecologists. Nutrition during pregnancy. ACOG Committee Opinion No. 804. *Obstet Gynecol.* 2020;135(1):e1–e17.
48. Institute of Medicine. *Dietary Reference Intakes for Energy, Carbohydrate, Fiber, Fat, Fatty Acids, Cholesterol, Protein, and Amino Acids.* Washington, DC: National Academies Press; 2006.
49. U.S. Food and Drug Administration / U.S. Environmental Protection Agency. Advice about eating fish: for those who might become or are pregnant, breastfeeding, and children ages 1–11 years. FDA/EPA; 2024.

### Anemia

50. National Institutes of Health, Office of Dietary Supplements. Iron fact sheet for health professionals. Available at: ods.od.nih.gov. Accessed March 2026.
51. World Health Organization. *Iron Deficiency Anaemia: Assessment, Prevention and Control — A Guide for Programme Managers.* Geneva: WHO; 2001.
