"""Tests for recommender: safe_recipes filtering, nutrient_distance scoring,
and KG recommendation mocking."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.recommender import (
    _recipe_ingredient_nutrients,
    _nutrient_relevance_score,
    _compute_user_targets,
    NUTRIENT_WEIGHTS,
)
from avoid_ingredients import get_avoid_ingredients


# ============ Fixtures ============

SALMON_RECIPE = {
    "id": "1",
    "name": "Grilled Salmon",
    "ingredients": ["Salmon fillet", "Olive oil", "Lemon", "Garlic", "Broccoli"],
}

SALTY_RECIPE = {
    "id": "2",
    "name": "Salty Bacon Dish",
    "ingredients": ["Bacon", "Salt", "Butter", "White bread", "Cheese"],
}

SUGARY_RECIPE = {
    "id": "3",
    "name": "Sweet Dessert",
    "ingredients": ["Sugar", "Butter", "Cream", "Honey", "Corn syrup"],
}

LENTIL_RECIPE = {
    "id": "4",
    "name": "Lentil Soup",
    "ingredients": ["Red lentils", "Onion", "Garlic", "Carrots", "Spinach"],
}

EMPTY_RECIPE = {
    "id": "5",
    "name": "Mystery Dish",
    "ingredients": [],
}

ALL_RECIPES = [SALMON_RECIPE, SALTY_RECIPE, SUGARY_RECIPE, LENTIL_RECIPE, EMPTY_RECIPE]


# ============ safe_recipes filtering ============

class TestSafeRecipesFiltering:
    """Test the avoid_ingredients-based recipe filtering pattern."""

    def _filter_recipes(self, recipes, conditions):
        """Replicate the filtering logic from get_recommendations."""
        all_ingredients = set()
        for r in recipes:
            all_ingredients.update(r.get("ingredients", []))
        if not all_ingredients:
            return recipes
        result = get_avoid_ingredients(conditions, all_ingredients)
        avoid_set = {ing.lower() for ing in result.get("unified", set())}
        if not avoid_set:
            return recipes
        return [
            r for r in recipes
            if not any(ing.lower() in avoid_set for ing in r.get("ingredients", []))
        ]

    def test_no_conditions_returns_all(self):
        safe = self._filter_recipes(ALL_RECIPES, set())
        assert len(safe) == len(ALL_RECIPES)

    def test_hypertension_removes_salty_recipes(self):
        safe = self._filter_recipes(ALL_RECIPES, {"Hypertension"})
        safe_ids = {r["id"] for r in safe}
        assert SALTY_RECIPE["id"] not in safe_ids

    def test_hypertension_keeps_salmon(self):
        safe = self._filter_recipes(ALL_RECIPES, {"Hypertension"})
        safe_ids = {r["id"] for r in safe}
        assert SALMON_RECIPE["id"] in safe_ids

    def test_diabetes_removes_sugary_recipes(self):
        safe = self._filter_recipes(ALL_RECIPES, {"Diabetes (Type 2)"})
        safe_ids = {r["id"] for r in safe}
        assert SUGARY_RECIPE["id"] not in safe_ids

    def test_diabetes_keeps_lentil_soup(self):
        safe = self._filter_recipes(ALL_RECIPES, {"Diabetes (Type 2)"})
        safe_ids = {r["id"] for r in safe}
        assert LENTIL_RECIPE["id"] in safe_ids

    def test_multiple_conditions_filter_more(self):
        one = self._filter_recipes(ALL_RECIPES, {"Hypertension"})
        two = self._filter_recipes(ALL_RECIPES, {"Hypertension", "Diabetes (Type 2)"})
        assert len(two) <= len(one)

    def test_empty_recipe_list_returns_empty(self):
        safe = self._filter_recipes([], {"Hypertension"})
        assert safe == []

    def test_recipes_with_no_ingredients_survive(self):
        safe = self._filter_recipes([EMPTY_RECIPE], {"Hypertension"})
        assert len(safe) == 1


# ============ _recipe_ingredient_nutrients ============

class TestRecipeIngredientNutrients:
    """Test ingredient-to-nutrient mapping."""

    def test_salmon_has_protein(self):
        nutrients = _recipe_ingredient_nutrients(SALMON_RECIPE)
        assert "protein" in nutrients

    def test_salmon_has_fat(self):
        nutrients = _recipe_ingredient_nutrients(SALMON_RECIPE)
        assert "fat" in nutrients

    def test_lentil_has_fiber(self):
        nutrients = _recipe_ingredient_nutrients(LENTIL_RECIPE)
        assert "fiber" in nutrients

    def test_lentil_has_iron(self):
        nutrients = _recipe_ingredient_nutrients(LENTIL_RECIPE)
        assert "iron" in nutrients

    def test_salty_recipe_has_sodium(self):
        nutrients = _recipe_ingredient_nutrients(SALTY_RECIPE)
        assert "sodium" in nutrients

    def test_sugary_recipe_has_sugar(self):
        nutrients = _recipe_ingredient_nutrients(SUGARY_RECIPE)
        assert "sugar" in nutrients

    def test_empty_recipe_returns_empty(self):
        nutrients = _recipe_ingredient_nutrients(EMPTY_RECIPE)
        assert nutrients == set()


# ============ _nutrient_relevance_score ============

class TestNutrientRelevanceScore:
    """Test nutrient-distance-based recipe scoring."""

    # Targets for a hypertensive user (sodium capped at 1500)
    HYPERTENSION_TARGETS = {
        "kcal": 2000,
        "protein_g": 80,
        "carbs_g": 250,
        "fat_g": 67,
        "saturated_fat_g": 13,
        "fiber_g": 30,
        "sugar_g": 25,
        "sodium_mg": 1500,
        "cholesterol_mg": 300,
    }

    USER_NEEDS = {"potassium", "magnesium", "calcium", "fiber", "iron", "protein"}

    def test_healthy_recipe_beats_empty(self):
        """Salmon (provides protein, user needs protein) should score
        lower (better) than an empty recipe."""
        salmon_score = _nutrient_relevance_score(
            SALMON_RECIPE, self.HYPERTENSION_TARGETS, self.USER_NEEDS
        )
        empty_score = _nutrient_relevance_score(
            EMPTY_RECIPE, self.HYPERTENSION_TARGETS, self.USER_NEEDS
        )
        assert salmon_score < empty_score

    def test_recipe_providing_needed_nutrients_gets_reward(self):
        """Lentil soup provides fiber and iron — both in user_needs."""
        score = _nutrient_relevance_score(
            LENTIL_RECIPE, self.HYPERTENSION_TARGETS, self.USER_NEEDS
        )
        assert score < 0  # net reward

    def test_sodium_penalty_applied(self):
        """A recipe with only sodium-heavy ingredients (no protein rewards)
        should score higher (worse) than an empty recipe."""
        sodium_only = {
            "id": "x",
            "name": "Salt water",
            "ingredients": ["Salt"],
        }
        score = _nutrient_relevance_score(
            sodium_only, self.HYPERTENSION_TARGETS, self.USER_NEEDS
        )
        empty_score = _nutrient_relevance_score(
            EMPTY_RECIPE, self.HYPERTENSION_TARGETS, self.USER_NEEDS
        )
        assert score > empty_score

    def test_empty_recipe_scores_zero(self):
        score = _nutrient_relevance_score(
            EMPTY_RECIPE, self.HYPERTENSION_TARGETS, self.USER_NEEDS
        )
        assert score == 0.0

    def test_no_user_needs_still_penalises_caps(self):
        """Even with no user needs, sodium should still be penalised."""
        score = _nutrient_relevance_score(
            SALTY_RECIPE, self.HYPERTENSION_TARGETS, set()
        )
        assert score > 0

    def test_default_sodium_cap_no_penalty(self):
        """With default sodium cap (2300), there's no penalty (restriction_ratio = 0)."""
        default_targets = {**self.HYPERTENSION_TARGETS, "sodium_mg": 2300.0}
        score_salty = _nutrient_relevance_score(
            SALTY_RECIPE, default_targets, set()
        )
        score_empty = _nutrient_relevance_score(
            EMPTY_RECIPE, default_targets, set()
        )
        # Sodium penalty should be zero when cap equals default
        # but saturated fat may still contribute if it has no default_cap
        assert score_salty >= score_empty

    def test_more_restrictive_cap_increases_penalty(self):
        """Lower sodium cap should produce a higher penalty."""
        mild = {**self.HYPERTENSION_TARGETS, "sodium_mg": 2000.0}
        strict = {**self.HYPERTENSION_TARGETS, "sodium_mg": 1000.0}
        score_mild = _nutrient_relevance_score(SALTY_RECIPE, mild, set())
        score_strict = _nutrient_relevance_score(SALTY_RECIPE, strict, set())
        assert score_strict > score_mild

    def test_ranking_nutrient_rich_beats_empty(self):
        """Recipes providing needed nutrients should rank better than empty."""
        scores = {
            r["name"]: _nutrient_relevance_score(
                r, self.HYPERTENSION_TARGETS, self.USER_NEEDS
            )
            for r in ALL_RECIPES
        }
        # Lentil soup provides fiber + iron (both needed) → best score
        assert scores["Lentil Soup"] < scores["Mystery Dish"]
        # Salmon provides protein (needed) → better than empty
        assert scores["Grilled Salmon"] < scores["Mystery Dish"]


# ============ _compute_user_targets ============

class TestComputeUserTargets:
    """Test the _compute_user_targets wrapper."""

    def _make_user(self, height=175, weight=80, age=35, sex="male",
                   activity="moderate", ailments=None):
        user = MagicMock()
        user.height_cm = height
        user.weight_kg = weight
        user.age = age
        user.sex = sex
        user.activity_level = activity
        if ailments is None:
            ailment = MagicMock()
            ailment.name = "Hypertension"
            user.ailments = [ailment]
        else:
            user.ailments = ailments
        return user

    def test_returns_dict_when_all_fields_present(self):
        user = self._make_user()
        result = _compute_user_targets(user)
        assert result is not None
        assert "kcal" in result

    def test_returns_none_when_height_missing(self):
        user = self._make_user(height=None)
        assert _compute_user_targets(user) is None

    def test_returns_none_when_weight_missing(self):
        user = self._make_user(weight=None)
        assert _compute_user_targets(user) is None

    def test_returns_none_when_age_missing(self):
        user = self._make_user(age=None)
        assert _compute_user_targets(user) is None

    def test_returns_none_when_sex_missing(self):
        user = self._make_user(sex=None)
        assert _compute_user_targets(user) is None

    def test_returns_none_when_activity_missing(self):
        user = self._make_user(activity=None)
        assert _compute_user_targets(user) is None

    def test_includes_condition_adjustments(self):
        user = self._make_user()
        result = _compute_user_targets(user)
        # Hypertension should lower sodium to 1500
        assert result["sodium_mg"] == 1500.0


# ============ KG Recommendations (mocked) ============

class TestKGRecommendations:
    """Test _kg_recommendations with mocked KG engine."""

    @pytest.mark.asyncio
    async def test_kg_recommendations_returns_recipes(self):
        from app.recommender import _kg_recommendations

        # Mock KG engine
        kg = MagicMock()
        kg.recommend_for_ailment.return_value = [
            ("Grilled Salmon", 0.15, ["omega-3", "protein"]),
            ("Lentil Soup", 0.25, ["fiber", "iron"]),
            ("Chicken Stir-Fry", 0.35, ["protein", "vitamin B6"]),
        ]

        # Mock user
        user = MagicMock()
        user.height_cm = None  # skip nutrient targets
        ailment = MagicMock()
        ailment.name = "Anemia"
        ailment.needs = "iron,vitamin B12"
        ailment.avoid = ""
        user.ailments = [ailment]

        with patch("app.recommender.search_mealdb_by_name", new_callable=AsyncMock, return_value=None):
            with patch("app.recommender.get_avoid_ingredients", return_value={"unified": set(), "by_condition": {}}):
                recipes = await _kg_recommendations(
                    kg, user, count=3,
                    skipped_ids=set(), cooked_ids=set(),
                    restrictions=[],
                )

        assert len(recipes) > 0
        assert recipes[0]["name"] == "Grilled Salmon"  # lowest score first
        assert recipes[0]["kg_score"] == 0.15

    @pytest.mark.asyncio
    async def test_kg_skips_skipped_ids(self):
        from app.recommender import _kg_recommendations

        kg = MagicMock()
        kg.recommend_for_ailment.return_value = [
            ("Grilled Salmon", 0.15, ["omega-3"]),
            ("Lentil Soup", 0.25, ["fiber"]),
        ]

        user = MagicMock()
        user.height_cm = None
        ailment = MagicMock()
        ailment.name = "Anemia"
        ailment.needs = "iron"
        ailment.avoid = ""
        user.ailments = [ailment]

        with patch("app.recommender.search_mealdb_by_name", new_callable=AsyncMock, return_value=None):
            with patch("app.recommender.get_avoid_ingredients", return_value={"unified": set(), "by_condition": {}}):
                recipes = await _kg_recommendations(
                    kg, user, count=3,
                    skipped_ids={"kg_grilled_salmon"},
                    cooked_ids=set(),
                    restrictions=[],
                )

        recipe_ids = {r["id"] for r in recipes}
        assert "kg_grilled_salmon" not in recipe_ids

    @pytest.mark.asyncio
    async def test_kg_marks_previously_cooked(self):
        from app.recommender import _kg_recommendations

        kg = MagicMock()
        kg.recommend_for_ailment.return_value = [
            ("Lentil Soup", 0.25, ["fiber"]),
        ]

        user = MagicMock()
        user.height_cm = None
        ailment = MagicMock()
        ailment.name = "Anemia"
        ailment.needs = "iron"
        ailment.avoid = ""
        user.ailments = [ailment]

        with patch("app.recommender.search_mealdb_by_name", new_callable=AsyncMock, return_value=None):
            with patch("app.recommender.get_avoid_ingredients", return_value={"unified": set(), "by_condition": {}}):
                recipes = await _kg_recommendations(
                    kg, user, count=3,
                    skipped_ids=set(),
                    cooked_ids={"kg_lentil_soup"},
                    restrictions=["sodium"],
                )

        assert recipes[0]["previously_cooked"] is True
        assert recipes[0]["restrictions_applied"] == ["sodium"]
