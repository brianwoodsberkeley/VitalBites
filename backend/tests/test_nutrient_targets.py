"""Tests for nutrient_targets.compute_nutrient_targets()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nutrient_targets import compute_nutrient_targets


EXPECTED_KEYS = {
    "kcal", "protein_g", "carbs_g", "fat_g", "saturated_fat_g",
    "fiber_g", "sugar_g", "sodium_mg", "cholesterol_mg",
}

BASE_PARAMS = {
    "height_cm": 175.0,
    "weight_kg": 80.0,
    "age": 35,
    "sex": "male",
    "activity_level": "moderate",
    "health_conditions": set(),
}


class TestComputeNutrientTargetsBaseline:
    """Baseline targets with no health conditions."""

    def test_returns_all_expected_keys(self):
        result = compute_nutrient_targets(**BASE_PARAMS)
        assert set(result.keys()) == EXPECTED_KEYS

    def test_all_values_are_numeric(self):
        result = compute_nutrient_targets(**BASE_PARAMS)
        for key, val in result.items():
            assert isinstance(val, (int, float)), f"{key} is {type(val)}"

    def test_all_values_are_positive(self):
        result = compute_nutrient_targets(**BASE_PARAMS)
        for key, val in result.items():
            assert val > 0, f"{key} = {val}"

    def test_kcal_is_reasonable(self):
        result = compute_nutrient_targets(**BASE_PARAMS)
        assert 1200 < result["kcal"] < 4000

    def test_protein_is_reasonable(self):
        result = compute_nutrient_targets(**BASE_PARAMS)
        assert 40 < result["protein_g"] < 300

    def test_baseline_sodium_is_2300(self):
        result = compute_nutrient_targets(**BASE_PARAMS)
        assert result["sodium_mg"] == 2300.0

    def test_baseline_cholesterol_is_300(self):
        result = compute_nutrient_targets(**BASE_PARAMS)
        assert result["cholesterol_mg"] == 300.0

    def test_carbs_at_least_130g(self):
        """IOM RDA minimum for brain glucose."""
        result = compute_nutrient_targets(**BASE_PARAMS)
        assert result["carbs_g"] >= 130.0


class TestSexDifferences:
    """Male vs female should produce different targets."""

    def test_male_higher_kcal_than_female(self):
        male = compute_nutrient_targets(**{**BASE_PARAMS, "sex": "male"})
        female = compute_nutrient_targets(**{**BASE_PARAMS, "sex": "female"})
        assert male["kcal"] > female["kcal"]

    def test_male_higher_fiber_than_female_under_50(self):
        male = compute_nutrient_targets(**{**BASE_PARAMS, "sex": "male", "age": 35})
        female = compute_nutrient_targets(**{**BASE_PARAMS, "sex": "female", "age": 35})
        assert male["fiber_g"] >= female["fiber_g"]

    def test_male_higher_sugar_cap_than_female(self):
        male = compute_nutrient_targets(**{**BASE_PARAMS, "sex": "male"})
        female = compute_nutrient_targets(**{**BASE_PARAMS, "sex": "female"})
        assert male["sugar_g"] >= female["sugar_g"]


class TestActivityLevels:
    """Higher activity level should increase caloric targets."""

    def test_sedentary_less_than_active(self):
        sed = compute_nutrient_targets(**{**BASE_PARAMS, "activity_level": "sedentary"})
        act = compute_nutrient_targets(**{**BASE_PARAMS, "activity_level": "active"})
        assert sed["kcal"] < act["kcal"]

    def test_all_activity_levels_produce_valid_output(self):
        for level in ["sedentary", "light", "moderate", "active", "very_active"]:
            result = compute_nutrient_targets(**{**BASE_PARAMS, "activity_level": level})
            assert result["kcal"] > 0


class TestHypertension:
    """Hypertension should lower sodium and sat fat."""

    def test_sodium_lowered_to_1500(self):
        result = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Hypertension"}}
        )
        assert result["sodium_mg"] == 1500.0

    def test_saturated_fat_lowered(self):
        baseline = compute_nutrient_targets(**BASE_PARAMS)
        hyp = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Hypertension"}}
        )
        assert hyp["saturated_fat_g"] < baseline["saturated_fat_g"]


class TestHeartDisease:
    """Heart Disease should lower cholesterol and sat fat."""

    def test_cholesterol_lowered_to_200(self):
        result = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Heart Disease"}}
        )
        assert result["cholesterol_mg"] == 200.0

    def test_sodium_lowered_to_1500(self):
        result = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Heart Disease"}}
        )
        assert result["sodium_mg"] == 1500.0


class TestDiabetes:
    """Diabetes should lower carbs and sugar."""

    def test_sugar_lowered_to_25(self):
        result = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Diabetes (Type 2)"}}
        )
        assert result["sugar_g"] == 25.0

    def test_carbs_lowered(self):
        baseline = compute_nutrient_targets(**BASE_PARAMS)
        diabetes = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Diabetes (Type 2)"}}
        )
        assert diabetes["carbs_g"] <= baseline["carbs_g"]


class TestKidneyDisease:
    """Kidney Disease should cap protein at 0.8 g/kg."""

    def test_protein_capped(self):
        result = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Kidney Disease"}}
        )
        # 0.8 g/kg * 80 kg = 64g, but AMDR 10% floor may raise it slightly
        # Should still be well below Weight Management's 1.4 g/kg = 112g
        assert result["protein_g"] < 80.0 * 1.0  # below 1.0 g/kg

    def test_kidney_overrides_weight_management_protein(self):
        """Kidney cap should override Weight Management's protein increase."""
        kidney_wm = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Kidney Disease", "Weight Management"}}
        )
        wm_only = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Weight Management"}}
        )
        assert kidney_wm["protein_g"] < wm_only["protein_g"]

    def test_sodium_lowered_to_2000(self):
        result = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Kidney Disease"}}
        )
        assert result["sodium_mg"] == 2000.0


class TestWeightManagement:
    """Weight Management should reduce kcal by 500."""

    def test_kcal_reduced_by_500(self):
        baseline = compute_nutrient_targets(**BASE_PARAMS)
        wm = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Weight Management"}}
        )
        assert abs((baseline["kcal"] - wm["kcal"]) - 500) < 2  # rounding

    def test_protein_increased(self):
        baseline = compute_nutrient_targets(**BASE_PARAMS)
        wm = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Weight Management"}}
        )
        assert wm["protein_g"] > baseline["protein_g"]


class TestConflictResolution:
    """Multiple conditions should resolve to most conservative values."""

    def test_sodium_takes_lowest(self):
        """Hypertension (1500) + Kidney Disease (2000) → 1500."""
        result = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Hypertension", "Kidney Disease"}}
        )
        assert result["sodium_mg"] == 1500.0

    def test_fiber_takes_highest(self):
        baseline = compute_nutrient_targets(**BASE_PARAMS)
        multi = compute_nutrient_targets(
            **{**BASE_PARAMS, "health_conditions": {"Hypertension", "Inflammation"}}
        )
        assert multi["fiber_g"] >= baseline["fiber_g"]

    def test_pregnancy_overrides_weight_management_calories(self):
        """Pregnancy surplus (+350) should override WM deficit (-500)."""
        preg_wm = compute_nutrient_targets(
            **{**BASE_PARAMS, "sex": "female",
               "health_conditions": {"Pregnancy Nutrition", "Weight Management"}}
        )
        baseline = compute_nutrient_targets(
            **{**BASE_PARAMS, "sex": "female"}
        )
        assert preg_wm["kcal"] > baseline["kcal"]
