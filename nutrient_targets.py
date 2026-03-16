"""
nutrient_targets.py
===================
Estimate daily nutrient-intake targets grounded in Dietary Reference
Intakes (DRIs), validated metabolic formulas, and condition-specific
clinical guidelines.

Avoidance lists are handled separately by avoid_ingredients.py.

Sources
-------
Mifflin-St Jeor (Am J Clin Nutr 1990); IOM 2002/2005 DRI tables;
AHA 2021 Dietary Guidance; ADA Standards of Care 2024-2026;
KDOQI 2020 / KDIGO 2024; ACOG; NHLBI Obesity Guidelines;
DASH trial (Sacks, NEJM 2001); Morton (Br J Sports Med 2018);
NPUAP/EPUAP 2019; ESPEN 2014; WHO 2015.
"""

from __future__ import annotations


def compute_nutrient_targets(
    height_cm: float,
    weight_kg: float,
    age: int,
    sex: str,
    activity_level: str,
    health_conditions: set[str],
) -> dict:
    """
    Estimate daily nutrient-intake targets based on DRIs and metabolic
    formulas, adjusted for health conditions.

    Parameters
    ----------
    height_cm : float
        Height in centimetres.
    weight_kg : float
        Body weight in kilograms.
    age : int
        Age in years.
    sex : str
        Biological sex — "male" or "female".
    activity_level : str
        One of "sedentary", "light", "moderate", "active", "very_active".
    health_conditions : set[str]
        Any subset of:
        "Hypertension", "Heart Disease", "Diabetes (Type 2)",
        "Thyroid Disorder", "Weight Management", "Kidney Disease",
        "Osteoporosis", "Anemia", "Immune Deficiency", "Inflammation",
        "Cognitive Decline", "Depression", "Fatigue", "Cramps",
        "Muscle Weakness", "Wound Healing", "Hair Loss",
        "Skin Conditions", "Pregnancy Nutrition".

    Returns
    -------
    dict
        Keys: kcal, protein_g, carbs_g, fat_g, saturated_fat_g,
              fiber_g, sugar_g, sodium_mg, cholesterol_mg.

    Notes
    -----
    Ingredient-level avoidance lists are produced by
    avoid_ingredients.get_avoid_ingredients() and are intentionally
    kept in a separate module.
    """

    # ── 1. BASAL METABOLIC RATE — Mifflin-St Jeor (1990) ─────────────────
    # Mifflin MD et al., Am J Clin Nutr 1990;51:241-7.
    # Validated as most accurate by Frankenfield et al., JADA 2005.
    # Male:   10 × wt + 6.25 × ht − 5 × age + 5
    # Female: 10 × wt + 6.25 × ht − 5 × age − 161

    if sex == "male":
        bmr = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age + 5.0
    else:
        bmr = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age - 161.0

    # ── 2. TOTAL DAILY ENERGY EXPENDITURE (TDEE) ─────────────────────────
    # PAL multipliers from FAO/WHO/UNU 2004 energy-requirements report.
    # Five-tier scale is the de-facto standard in clinical nutrition practice.

    pal_multipliers = {
        "sedentary":   1.2,    # little / no exercise, desk job
        "light":       1.375,  # light exercise 1-3 days/week
        "moderate":    1.55,   # moderate exercise 3-5 days/week
        "active":      1.725,  # hard exercise 6-7 days/week
        "very_active": 1.9,    # very hard exercise / physical job
    }

    tdee = bmr * pal_multipliers[activity_level]

    # ── 3. BASELINE MACRONUTRIENT TARGETS (IOM 2002/2005 DRI) ────────────
    # AMDRs: Protein 10-35 % | Carbs 45-65 % | Fat 20-35 % of kcal
    # Default split: 50 % carb / 20 % protein / 30 % fat
    # Protein floor: RDA = 0.8 g/kg (IOM 2005, nitrogen-balance studies)

    protein_pct      = 0.20   # noqa: F841  (kept for readability)
    carb_pct         = 0.50
    protein_g_per_kg = 0.8    # IOM RDA

    # Saturated fat — AHA / 2020-2025 Dietary Guidelines for Americans
    # General population: <10 % kcal (DGA); CVD goal: <7 % (TLC / AHA)
    sat_fat_pct = 0.10

    # Fiber — IOM 2005 Adequate Intake (14 g per 1,000 kcal baseline)
    if sex == "male":
        fiber_g = 38.0 if age <= 50 else 30.0
    else:
        fiber_g = 25.0 if age <= 50 else 21.0

    # Added sugar — WHO 2015 strong rec <10 % kcal; AHA 2009 absolute caps
    # Women ≤25 g/day, men ≤36 g/day  (Johnson, Circ 2009)
    sugar_g = 36.0 if sex == "male" else 25.0

    # Sodium — 2020-2025 DGA upper limit 2,300 mg; AHA ideal 1,500 mg
    sodium_mg = 2300.0

    # Cholesterol — <300 mg general cap (NCEP ATP III)
    cholesterol_mg = 300.0

    # ── 4. CONDITION-SPECIFIC ADJUSTMENT ACCUMULATORS ────────────────────

    protein_g_per_kg_requests: list[float] = []
    carb_pct_requests:         list[float] = []
    fiber_g_requests:          list[float] = []
    sugar_g_requests:          list[float] = []
    sodium_requests:           list[float] = []
    sat_fat_pct_requests:      list[float] = []
    cholesterol_requests:      list[float] = []
    kcal_deltas:               list[float] = []

    kidney_disease_present = "Kidney Disease"      in health_conditions
    pregnancy_present      = "Pregnancy Nutrition" in health_conditions
    weight_mgmt_present    = "Weight Management"   in health_conditions

    # ── HYPERTENSION ──────────────────────────────────────────────────────
    # DASH diet: Sacks FM et al., NEJM 2001; NHLBI DASH Eating Plan;
    # 2025 AHA/ACC Hypertension Guidelines.
    if "Hypertension" in health_conditions:
        sodium_requests.append(1500.0)      # AHA / DASH-Sodium target
        sat_fat_pct_requests.append(0.06)   # DASH trial ~6 % of kcal
        fiber_g_requests.append(30.0)       # DASH naturally ≥30 g

    # ── HEART DISEASE ─────────────────────────────────────────────────────
    # AHA 2021 Dietary Guidance (Lichtenstein, Circ 2021);
    # NCEP ATP III TLC Diet; AHA 2017 Presidential Advisory (Sacks, Circ 2017).
    if "Heart Disease" in health_conditions:
        sat_fat_pct_requests.append(0.07)   # <7 % TLC / AHA for LDL-C
        cholesterol_requests.append(200.0)  # <200 mg (NCEP ATP III; NLA)
        sodium_requests.append(1500.0)      # AHA ideal target
        fiber_g_requests.append(25.0)       # TLC 20-30 g; soluble 10-25 g

    # ── DIABETES (TYPE 2) ─────────────────────────────────────────────────
    # ADA Standards of Care 2024-2026, Section 5 (Recs 5.12-5.31);
    # ADA/EASD Nutrition Consensus (Evert, Diabetes Care 2019).
    if "Diabetes (Type 2)" in health_conditions:
        carb_pct_requests.append(0.45)      # practical lower target
        sugar_g_requests.append(25.0)       # minimise added sugars
        fiber_g_requests.append(28.0)       # ADA Rec 5.24: ≥14 g/1 000 kcal
        sodium_requests.append(2300.0)      # ADA Rec 5.20

    # ── KIDNEY DISEASE (CKD stages 3-5, non-dialysis) ─────────────────────
    # KDOQI 2020 Update (Ikizler, AJKD 2020); KDIGO 2024 Rec 3.3.1.1.
    if kidney_disease_present:
        protein_g_per_kg_requests.append(0.8)   # KDIGO 2024
        sodium_requests.append(2000.0)           # KDOQI 2020 conservative cap

    # ── WEIGHT MANAGEMENT ─────────────────────────────────────────────────
    # NHLBI Obesity Guidelines; Morton, Br J Sports Med 2018.
    if weight_mgmt_present:
        kcal_deltas.append(-500.0)               # ~0.5 kg/week deficit (NHLBI)
        if not kidney_disease_present:
            protein_g_per_kg_requests.append(1.4)   # 1.2-1.6 g/kg midpoint
        carb_pct_requests.append(0.40)
        fiber_g_requests.append(30.0)
        sugar_g_requests.append(25.0)

    # ── OSTEOPOROSIS ──────────────────────────────────────────────────────
    # Bone Health & Osteoporosis Foundation; IOM 2011 DRIs.
    # Excess sodium increases urinary calcium excretion — cap at DGA limit.
    if "Osteoporosis" in health_conditions:
        sodium_requests.append(2300.0)

    # ── IMMUNE DEFICIENCY ─────────────────────────────────────────────────
    # NIH ODS; Gombart, Nutrients 2020.
    if "Immune Deficiency" in health_conditions:
        sugar_g_requests.append(25.0)       # excess sugar impairs neutrophils
        if not kidney_disease_present:
            protein_g_per_kg_requests.append(1.0)

    # ── INFLAMMATION ──────────────────────────────────────────────────────
    # Koelman, Adv Nutr 2022; AHA 2021; Calder, AJCN 2006.
    if "Inflammation" in health_conditions:
        sat_fat_pct_requests.append(0.08)
        fiber_g_requests.append(30.0)

    # ── COGNITIVE DECLINE ─────────────────────────────────────────────────
    # MIND diet: Morris MC et al., Alzheimers Dement 2015; NEJM 2023 RCT.
    if "Cognitive Decline" in health_conditions:
        sat_fat_pct_requests.append(0.08)

    # ── DEPRESSION ────────────────────────────────────────────────────────
    # SMILES Trial (Jacka, BMC Med 2017); WFSBP/CANMAT 2022.
    if "Depression" in health_conditions:
        sugar_g_requests.append(25.0)

    # ── MUSCLE WEAKNESS ───────────────────────────────────────────────────
    # ESPEN (Bauer, JAMDA 2013); EWGSOP2 2019 sarcopenia criteria.
    if "Muscle Weakness" in health_conditions:
        if not kidney_disease_present:
            protein_g_per_kg_requests.append(1.4)  # 1.2-1.6 g/kg midpoint

    # ── WOUND HEALING ─────────────────────────────────────────────────────
    # NPUAP/EPUAP/PPPIA 2019; ASPEN.
    if "Wound Healing" in health_conditions:
        if not kidney_disease_present:
            protein_g_per_kg_requests.append(1.5)  # NPUAP/EPUAP 2019

    # ── PREGNANCY NUTRITION ───────────────────────────────────────────────
    # ACOG; IOM 2006 energy requirements (+340 2nd tri, +452 3rd tri);
    # NIH ODS Pregnancy fact sheet.
    if pregnancy_present:
        kcal_deltas.append(350.0)           # average 2nd/3rd trimester
        if not kidney_disease_present:
            protein_g_per_kg_requests.append(
                protein_g_per_kg + (25.0 / weight_kg)  # +25 g above baseline
            )
        fiber_g_requests.append(28.0)       # IOM AI for pregnancy

    # Conditions that affect only avoidance lists — no macro targets modified:
    # Anemia, Thyroid Disorder, Fatigue, Cramps, Hair Loss, Skin Conditions
    # → see avoid_ingredients.py

    # ── 5. RESOLVE CONFLICTS — MOST CONSERVATIVE VALUE ───────────────────
    # Protein : Kidney Disease restriction overrides all increases
    if kidney_disease_present and protein_g_per_kg_requests:
        protein_g_per_kg = min(protein_g_per_kg_requests)
    elif protein_g_per_kg_requests:
        protein_g_per_kg = max(protein_g_per_kg_requests)

    if carb_pct_requests:
        carb_pct = min(carb_pct_requests)

    if sat_fat_pct_requests:
        sat_fat_pct = min(sat_fat_pct_requests)

    if fiber_g_requests:
        fiber_g = max(fiber_g, max(fiber_g_requests))

    if sugar_g_requests:
        sugar_g = min(sugar_g, min(sugar_g_requests))

    if sodium_requests:
        sodium_mg = min(sodium_requests)

    if cholesterol_requests:
        cholesterol_mg = min(cholesterol_requests)

    # Pregnancy surplus beats weight-loss deficit
    if pregnancy_present and weight_mgmt_present:
        kcal_adjustment = 350.0
    else:
        kcal_adjustment = sum(kcal_deltas)

    # ── 6. COMPUTE FINAL NUMERIC TARGETS ─────────────────────────────────

    kcal = round(tdee + kcal_adjustment)

    # Protein — clamp to AMDR 10-35 % of kcal
    protein_g = round(protein_g_per_kg * weight_kg, 1)
    protein_kcal_pct = (protein_g * 4.0) / kcal if kcal > 0 else 0
    if protein_kcal_pct > 0.35:
        protein_g = round(0.35 * kcal / 4.0, 1)
    elif protein_kcal_pct < 0.10:
        protein_g = round(0.10 * kcal / 4.0, 1)

    # Carbohydrates — floor at IOM RDA of 130 g/day (brain glucose requirement)
    carbs_g = round(carb_pct * kcal / 4.0, 1)
    if carbs_g < 130.0:
        carbs_g = 130.0

    # Fat — allocate remaining kcal; enforce AMDR 20 % floor
    remaining_kcal = kcal - (protein_g * 4.0) - (carbs_g * 4.0)
    fat_g = round(max(remaining_kcal, 0) / 9.0, 1)
    fat_floor = round(0.20 * kcal / 9.0, 1)
    if fat_g < fat_floor:
        fat_g = fat_floor

    saturated_fat_g = round(sat_fat_pct * kcal / 9.0, 1)

    return {
        "kcal":            kcal,
        "protein_g":       protein_g,
        "carbs_g":         carbs_g,
        "fat_g":           fat_g,
        "saturated_fat_g": saturated_fat_g,
        "fiber_g":         round(fiber_g, 1),
        "sugar_g":         round(sugar_g, 1),
        "sodium_mg":       round(sodium_mg, 1),
        "cholesterol_mg":  round(cholesterol_mg, 1),
    }
