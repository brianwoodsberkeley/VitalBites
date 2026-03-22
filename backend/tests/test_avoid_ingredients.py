"""Tests for avoid_ingredients.get_avoid_ingredients()."""

import sys
from pathlib import Path

# Add repo root so we can import the module directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avoid_ingredients import get_avoid_ingredients


class TestGetAvoidIngredients:
    """Tests for the get_avoid_ingredients function."""

    SAMPLE_INGREDIENTS = [
        "Salmon fillet", "Olive oil", "Lemon", "Garlic", "Broccoli",
        "Salt", "Soy sauce", "Butter", "Sugar", "White bread",
        "Bacon", "Beer", "Vodka", "Corn syrup", "Margarine",
        "Whole milk", "Coconut oil", "Honey", "Rice", "Spinach",
        "Avocado", "Banana", "Tofu", "Kelp", "Raw egg whites",
        "Coffee", "Tea", "Liver", "Peanut butter",
    ]

    def test_returns_dict_with_expected_keys(self):
        result = get_avoid_ingredients({"Hypertension"}, self.SAMPLE_INGREDIENTS)
        assert "unified" in result
        assert "by_condition" in result
        assert isinstance(result["unified"], set)
        assert isinstance(result["by_condition"], dict)

    def test_empty_conditions_returns_empty(self):
        result = get_avoid_ingredients(set(), self.SAMPLE_INGREDIENTS)
        assert result["unified"] == set()
        assert result["by_condition"] == {}

    def test_empty_ingredient_list_returns_empty(self):
        result = get_avoid_ingredients({"Hypertension"}, [])
        assert result["unified"] == set()

    def test_hypertension_flags_salt(self):
        result = get_avoid_ingredients({"Hypertension"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "salt" in unified_lower

    def test_hypertension_flags_soy_sauce(self):
        result = get_avoid_ingredients({"Hypertension"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "soy sauce" in unified_lower

    def test_hypertension_does_not_flag_salmon(self):
        result = get_avoid_ingredients({"Hypertension"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "salmon fillet" not in unified_lower

    def test_diabetes_flags_corn_syrup_in_unified(self):
        result = get_avoid_ingredients({"Diabetes (Type 2)"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "corn syrup" in unified_lower

    def test_diabetes_flags_white_sugar(self):
        result = get_avoid_ingredients({"Diabetes (Type 2)"}, ["White sugar", "Spinach"])
        unified_lower = {x.lower() for x in result["unified"]}
        assert "white sugar" in unified_lower

    def test_diabetes_flags_corn_syrup(self):
        result = get_avoid_ingredients({"Diabetes (Type 2)"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "corn syrup" in unified_lower

    def test_diabetes_does_not_flag_spinach(self):
        result = get_avoid_ingredients({"Diabetes (Type 2)"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "spinach" not in unified_lower

    def test_heart_disease_flags_butter(self):
        result = get_avoid_ingredients({"Heart Disease"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "butter" in unified_lower

    def test_thyroid_flags_kelp(self):
        result = get_avoid_ingredients({"Thyroid Disorder"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "kelp" in unified_lower

    def test_thyroid_flags_tofu(self):
        result = get_avoid_ingredients({"Thyroid Disorder"}, self.SAMPLE_INGREDIENTS)
        unified_lower = {x.lower() for x in result["unified"]}
        assert "tofu" in unified_lower

    def test_multiple_conditions_union(self):
        result = get_avoid_ingredients(
            {"Hypertension", "Diabetes (Type 2)"},
            self.SAMPLE_INGREDIENTS,
        )
        unified_lower = {x.lower() for x in result["unified"]}
        # Should include flags from both conditions
        assert "salt" in unified_lower
        assert "corn syrup" in unified_lower

    def test_by_condition_has_per_condition_keys(self):
        conditions = {"Hypertension", "Diabetes (Type 2)"}
        result = get_avoid_ingredients(conditions, self.SAMPLE_INGREDIENTS)
        assert "Hypertension" in result["by_condition"]
        assert "Diabetes (Type 2)" in result["by_condition"]

    def test_by_condition_lists_are_sorted(self):
        result = get_avoid_ingredients({"Hypertension"}, self.SAMPLE_INGREDIENTS)
        hyp_list = result["by_condition"].get("Hypertension", [])
        assert hyp_list == sorted(hyp_list)

    def test_unified_is_superset_of_all_condition_lists(self):
        conditions = {"Hypertension", "Heart Disease", "Diabetes (Type 2)"}
        result = get_avoid_ingredients(conditions, self.SAMPLE_INGREDIENTS)
        for condition, flagged in result["by_condition"].items():
            for ing in flagged:
                assert ing in result["unified"]

    def test_exclude_patterns_prevent_false_positives(self):
        """Peanut butter should not be flagged by a butter rule."""
        result = get_avoid_ingredients(
            {"Heart Disease"},
            ["Peanut butter", "Butter"],
        )
        unified_lower = {x.lower() for x in result["unified"]}
        assert "butter" in unified_lower
        assert "peanut butter" not in unified_lower

    def test_unknown_condition_returns_empty_for_that_condition(self):
        result = get_avoid_ingredients(
            {"Nonexistent Condition"},
            self.SAMPLE_INGREDIENTS,
        )
        assert result["by_condition"].get("Nonexistent Condition", []) == []
        assert result["unified"] == set()
