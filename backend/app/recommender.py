import httpx
import random
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session
from .models import User, RecipeFeedback
from avoid_ingredients import get_avoid_ingredients
from nutrient_targets import compute_nutrient_targets

# TheMealDB API - free, no API key required
MEALDB_API_BASE = "https://www.themealdb.com/api/json/v1/1"

# ============ KG Model Singleton ============

# Add repo root to sys.path so we can import train_and_infer
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/
_REPO_ROOT = _BACKEND_DIR.parent                       # VitalBites/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default model directory: backend/models/
MODELS_DIR = os.environ.get("VITALBITES_MODELS_DIR", str(_BACKEND_DIR / "models"))

# Cached KG inference engine (loaded once, reused across requests)
_kg_engine = None
_kg_load_attempted = False


def get_kg_engine():
    """
    Lazy-load and cache the KnowledgeGraphInference engine.
    Returns None if model files are not present.
    """
    global _kg_engine, _kg_load_attempted

    if _kg_load_attempted:
        return _kg_engine

    _kg_load_attempted = True

    # Check if required model files exist
    required_files = [
        "entity_to_id.json",
        "relation_to_id.json",
        "entity_embeddings.npy",
        "relation_embeddings.npy",
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print(f"[recommender] KG model files missing from {MODELS_DIR}: {missing}")
        print(f"[recommender] Falling back to TheMealDB/mock recommendations.")
        return None

    config_path = os.path.join(MODELS_DIR, "mined_config.json")
    if not os.path.exists(config_path):
        # Also check repo root as fallback
        config_path = str(_REPO_ROOT / "mined_config.json")
    if not os.path.exists(config_path):
        config_path = "mined_config.json"  # let EntityCatalog handle the missing file

    try:
        from train_and_infer import KnowledgeGraphInference
        print(f"[recommender] Loading KG model from {MODELS_DIR}...")
        _kg_engine = KnowledgeGraphInference(model_dir=MODELS_DIR, config_path=config_path)
        print(f"[recommender] KG model loaded successfully.")
    except Exception as e:
        print(f"[recommender] Failed to load KG model: {e}")
        _kg_engine = None

    return _kg_engine


# ============ TheMealDB Search ============

async def search_mealdb_by_name(name: str) -> Optional[dict]:
    """Search TheMealDB for a recipe by name and return formatted recipe dict."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{MEALDB_API_BASE}/search.php", params={"s": name})
            if response.status_code == 200:
                data = response.json()
                if data.get("meals"):
                    meal = data["meals"][0]
                    ingredients = []
                    for i in range(1, 21):
                        ingredient = meal.get(f"strIngredient{i}")
                        if ingredient and ingredient.strip():
                            ingredients.append(ingredient)
                    return {
                        "id": meal["idMeal"],
                        "name": meal["strMeal"],
                        "image": meal["strMealThumb"],
                        "category": meal["strCategory"],
                        "area": meal.get("strArea", ""),
                        "instructions": meal["strInstructions"],
                        "ingredients": ingredients,
                        "source": meal.get("strSource", ""),
                        "youtube": meal.get("strYoutube", ""),
                    }
    except Exception as e:
        print(f"[recommender] MealDB search failed for '{name}': {e}")
    return None


# ============ Fallback: TheMealDB random + mock ============

MOCK_RECIPES = [
    {
        "id": "mock_1",
        "name": "Grilled Salmon with Vegetables",
        "image": "https://via.placeholder.com/300x200?text=Grilled+Salmon",
        "category": "Seafood",
        "instructions": "Season salmon with herbs. Grill for 4-5 minutes per side. Serve with steamed vegetables.",
        "ingredients": ["Salmon fillet", "Olive oil", "Lemon", "Garlic", "Broccoli", "Asparagus"]
    },
    {
        "id": "mock_2",
        "name": "Quinoa Buddha Bowl",
        "image": "https://via.placeholder.com/300x200?text=Buddha+Bowl",
        "category": "Vegetarian",
        "instructions": "Cook quinoa. Arrange with roasted chickpeas, avocado, and vegetables. Drizzle with tahini.",
        "ingredients": ["Quinoa", "Chickpeas", "Avocado", "Cucumber", "Tomatoes", "Tahini"]
    },
    {
        "id": "mock_3",
        "name": "Chicken Stir-Fry",
        "image": "https://via.placeholder.com/300x200?text=Chicken+Stir+Fry",
        "category": "Chicken",
        "instructions": "Slice chicken and vegetables. Stir-fry in hot wok with soy sauce and ginger.",
        "ingredients": ["Chicken breast", "Bell peppers", "Broccoli", "Soy sauce", "Ginger", "Garlic"]
    },
    {
        "id": "mock_4",
        "name": "Mediterranean Salad",
        "image": "https://via.placeholder.com/300x200?text=Mediterranean+Salad",
        "category": "Salad",
        "instructions": "Combine fresh vegetables with feta cheese. Dress with olive oil and lemon.",
        "ingredients": ["Cucumber", "Tomatoes", "Red onion", "Feta cheese", "Olives", "Olive oil"]
    },
    {
        "id": "mock_5",
        "name": "Lentil Soup",
        "image": "https://via.placeholder.com/300x200?text=Lentil+Soup",
        "category": "Soup",
        "instructions": "Saut\u00e9 onions and garlic. Add lentils and broth. Simmer until tender.",
        "ingredients": ["Red lentils", "Onion", "Garlic", "Carrots", "Vegetable broth", "Cumin"]
    },
    {
        "id": "mock_6",
        "name": "Baked Cod with Herbs",
        "image": "https://via.placeholder.com/300x200?text=Baked+Cod",
        "category": "Seafood",
        "instructions": "Season cod with herbs and lemon. Bake at 400\u00b0F for 15-20 minutes.",
        "ingredients": ["Cod fillet", "Lemon", "Dill", "Parsley", "Olive oil", "Garlic"]
    },
    {
        "id": "mock_7",
        "name": "Turkey Lettuce Wraps",
        "image": "https://via.placeholder.com/300x200?text=Lettuce+Wraps",
        "category": "Poultry",
        "instructions": "Brown ground turkey with Asian seasonings. Serve in crisp lettuce cups.",
        "ingredients": ["Ground turkey", "Lettuce", "Water chestnuts", "Soy sauce", "Ginger", "Green onions"]
    },
    {
        "id": "mock_8",
        "name": "Vegetable Curry",
        "image": "https://via.placeholder.com/300x200?text=Vegetable+Curry",
        "category": "Vegetarian",
        "instructions": "Saut\u00e9 vegetables in curry paste. Add coconut milk and simmer.",
        "ingredients": ["Sweet potato", "Chickpeas", "Spinach", "Coconut milk", "Curry paste", "Rice"]
    },
    {
        "id": "mock_9",
        "name": "Grilled Chicken Salad",
        "image": "https://via.placeholder.com/300x200?text=Chicken+Salad",
        "category": "Salad",
        "instructions": "Grill seasoned chicken breast. Slice and serve over mixed greens.",
        "ingredients": ["Chicken breast", "Mixed greens", "Cherry tomatoes", "Cucumber", "Balsamic vinegar"]
    },
    {
        "id": "mock_10",
        "name": "Shrimp and Zucchini Noodles",
        "image": "https://via.placeholder.com/300x200?text=Zoodles",
        "category": "Seafood",
        "instructions": "Spiralize zucchini. Saut\u00e9 shrimp with garlic. Toss together with olive oil.",
        "ingredients": ["Shrimp", "Zucchini", "Garlic", "Cherry tomatoes", "Olive oil", "Basil"]
    },
    {
        "id": "mock_11",
        "name": "Black Bean Tacos",
        "image": "https://via.placeholder.com/300x200?text=Bean+Tacos",
        "category": "Mexican",
        "instructions": "Season black beans with cumin. Serve in corn tortillas with fresh salsa.",
        "ingredients": ["Black beans", "Corn tortillas", "Avocado", "Salsa", "Cilantro", "Lime"]
    },
    {
        "id": "mock_12",
        "name": "Oatmeal with Berries",
        "image": "https://via.placeholder.com/300x200?text=Oatmeal",
        "category": "Breakfast",
        "instructions": "Cook oats with almond milk. Top with fresh berries and a drizzle of honey.",
        "ingredients": ["Rolled oats", "Almond milk", "Blueberries", "Strawberries", "Honey", "Cinnamon"]
    },
    {
        "id": "mock_13",
        "name": "Stuffed Bell Peppers",
        "image": "https://via.placeholder.com/300x200?text=Stuffed+Peppers",
        "category": "Main",
        "instructions": "Fill peppers with seasoned ground meat and rice. Bake until tender.",
        "ingredients": ["Bell peppers", "Ground beef", "Rice", "Tomato sauce", "Onion", "Italian herbs"]
    },
    {
        "id": "mock_14",
        "name": "Greek Yogurt Parfait",
        "image": "https://via.placeholder.com/300x200?text=Yogurt+Parfait",
        "category": "Breakfast",
        "instructions": "Layer yogurt with granola and fresh fruit. Drizzle with honey.",
        "ingredients": ["Greek yogurt", "Granola", "Mixed berries", "Honey", "Almonds"]
    },
    {
        "id": "mock_15",
        "name": "Tomato Basil Soup",
        "image": "https://via.placeholder.com/300x200?text=Tomato+Soup",
        "category": "Soup",
        "instructions": "Roast tomatoes with garlic. Blend until smooth. Add fresh basil.",
        "ingredients": ["Tomatoes", "Garlic", "Onion", "Fresh basil", "Vegetable broth", "Olive oil"]
    }
]


async def fetch_recipes_from_api(count: int = 10) -> List[dict]:
    """Fetch random recipes from TheMealDB API"""
    recipes = []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for _ in range(count):
                response = await client.get(f"{MEALDB_API_BASE}/random.php")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("meals"):
                        meal = data["meals"][0]

                        # Extract ingredients
                        ingredients = []
                        for i in range(1, 21):
                            ingredient = meal.get(f"strIngredient{i}")
                            if ingredient and ingredient.strip():
                                ingredients.append(ingredient)

                        recipes.append({
                            "id": meal["idMeal"],
                            "name": meal["strMeal"],
                            "image": meal["strMealThumb"],
                            "category": meal["strCategory"],
                            "area": meal.get("strArea", ""),
                            "instructions": meal["strInstructions"],
                            "ingredients": ingredients,
                            "source": meal.get("strSource", ""),
                            "youtube": meal.get("strYoutube", "")
                        })
    except Exception as e:
        print(f"Error fetching from API: {e}")

    return recipes


def get_mock_recipes(count: int = 10) -> List[dict]:
    """Get random mock recipes as fallback"""
    return random.sample(MOCK_RECIPES, min(count, len(MOCK_RECIPES)))


# ============ Nutrient-based ranking ============

# Map ingredients to the macro/micro nutrients they provide.
# Keys are lowercase substrings matched against ingredient names.
INGREDIENT_NUTRIENT_MAP = {
    "salmon":   {"protein", "fat", "polyunsaturated fat"},
    "tuna":     {"protein", "fat"},
    "cod":      {"protein"},
    "shrimp":   {"protein"},
    "sardine":  {"protein", "fat", "calcium"},
    "mackerel": {"protein", "fat"},
    "chicken":  {"protein"},
    "turkey":   {"protein"},
    "beef":     {"protein", "iron"},
    "lamb":     {"protein", "iron"},
    "pork":     {"protein"},
    "egg":      {"protein", "fat", "cholesterol"},
    "tofu":     {"protein", "calcium", "iron"},
    "lentil":   {"protein", "fiber", "iron"},
    "chickpea": {"protein", "fiber", "iron"},
    "bean":     {"protein", "fiber", "iron"},
    "pea":      {"protein", "fiber"},
    "quinoa":   {"protein", "fiber", "carbohydrate"},
    "oat":      {"fiber", "carbohydrate", "iron"},
    "rice":     {"carbohydrate"},
    "pasta":    {"carbohydrate"},
    "bread":    {"carbohydrate", "fiber"},
    "potato":   {"carbohydrate", "potassium", "fiber"},
    "sweet potato": {"carbohydrate", "fiber", "potassium"},
    "corn":     {"carbohydrate", "fiber"},
    "avocado":  {"fat", "fiber", "potassium"},
    "olive oil": {"fat"},
    "coconut":  {"fat"},
    "butter":   {"fat", "saturated fat", "cholesterol"},
    "cream":    {"fat", "saturated fat"},
    "cheese":   {"protein", "fat", "saturated fat", "calcium", "sodium"},
    "milk":     {"protein", "calcium"},
    "yogurt":   {"protein", "calcium"},
    "almond":   {"fat", "fiber", "magnesium"},
    "walnut":   {"fat", "fiber"},
    "peanut":   {"protein", "fat"},
    "spinach":  {"iron", "fiber", "magnesium"},
    "kale":     {"fiber", "calcium", "iron"},
    "broccoli": {"fiber", "vitamin C"},
    "carrot":   {"fiber", "potassium"},
    "tomato":   {"potassium", "vitamin C"},
    "onion":    {"fiber"},
    "garlic":   set(),
    "ginger":   set(),
    "lemon":    {"vitamin C"},
    "honey":    {"sugar"},
    "sugar":    {"sugar"},
    "soy sauce": {"sodium"},
    "salt":     {"sodium"},
}

# Nutrient target keys → ingredient nutrient tags that contribute to them.
TARGET_TO_INGREDIENT_NUTRIENTS = {
    "protein_g":       {"protein"},
    "carbs_g":         {"carbohydrate"},
    "fat_g":           {"fat"},
    "fiber_g":         {"fiber"},
    "sodium_mg":       {"sodium"},
    "cholesterol_mg":  {"cholesterol"},
    "saturated_fat_g": {"saturated fat"},
    "sugar_g":         {"sugar"},
}

# Weights for nutrient distance scoring — how important each target is
# for ranking. Higher weight = more influence on ranking.
NUTRIENT_WEIGHTS = {
    "protein_g":       1.0,
    "carbs_g":         0.8,
    "fat_g":           0.6,
    "fiber_g":         1.0,
    "sodium_mg":       1.2,  # caps are clinically important
    "cholesterol_mg":  0.8,
    "saturated_fat_g": 1.0,
    "sugar_g":         1.0,
}


def _recipe_ingredient_nutrients(recipe: dict) -> set:
    """Derive the set of nutrient tags a recipe provides from its ingredients."""
    nutrients = set()
    for ing in recipe.get("ingredients", []):
        ing_lower = ing.lower()
        for key, tags in INGREDIENT_NUTRIENT_MAP.items():
            if key in ing_lower:
                nutrients.update(tags)
    return nutrients


def _nutrient_relevance_score(recipe: dict, targets: dict, user_needs: set) -> float:
    """
    Score a recipe by how well its ingredient-derived nutrients align
    with the user's nutrient targets.  Lower score = better match.

    Recipes that provide nutrients the user needs (based on ailment 'needs')
    are ranked higher. Recipes containing nutrients that should be limited
    (caps in targets) are penalised.
    """
    recipe_nutrients = _recipe_ingredient_nutrients(recipe)

    score = 0.0

    # Reward: recipe provides nutrients the user needs
    for nutrient_tag in recipe_nutrients:
        if nutrient_tag in user_needs:
            score -= 1.0  # lower is better

    # Penalise: recipe is high in nutrients that have low caps
    # (sodium, saturated fat, sugar, cholesterol)
    cap_nutrients = {
        "sodium":        ("sodium_mg",       2300.0),
        "saturated fat": ("saturated_fat_g", None),
        "sugar":         ("sugar_g",         None),
        "cholesterol":   ("cholesterol_mg",  300.0),
    }
    for nutrient_tag, (target_key, default_cap) in cap_nutrients.items():
        if nutrient_tag in recipe_nutrients:
            target_val = targets.get(target_key, default_cap)
            if target_val is not None and default_cap is not None:
                # The more restrictive the cap vs default, the bigger the penalty
                restriction_ratio = 1.0 - (target_val / default_cap)
                if restriction_ratio > 0:
                    score += NUTRIENT_WEIGHTS.get(target_key, 1.0) * restriction_ratio

    return score


def _compute_user_targets(user: "User") -> Optional[dict]:
    """Compute nutrient targets if the user has biometric data."""
    if not all([user.height_cm, user.weight_kg, user.age, user.sex, user.activity_level]):
        return None
    try:
        condition_names = {a.name for a in user.ailments}
        return compute_nutrient_targets(
            height_cm=user.height_cm,
            weight_kg=user.weight_kg,
            age=user.age,
            sex=user.sex,
            activity_level=user.activity_level,
            health_conditions=condition_names,
        )
    except Exception as e:
        print(f"[recommender] compute_nutrient_targets failed: {e}")
        return None


# ============ Main Recommendation Logic ============

async def get_recommendations(
    user: User,
    db: Session,
    count: int = 10
) -> List[dict]:
    """
    Get personalized recipe recommendations.
    Uses KG model when available, falls back to TheMealDB/mock.
    """

    # Get user's avoid list
    restrictions = []
    for ailment in user.ailments:
        if ailment.avoid:
            restrictions.extend(ailment.avoid.split(','))
    restrictions = list(set(r for r in restrictions if r))

    # Get previously skipped recipe IDs
    skipped_feedback = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == user.id,
        RecipeFeedback.skipped == True
    ).all()
    skipped_ids = {f.recipe_id for f in skipped_feedback}

    # Get previously cooked recipe IDs
    cooked_feedback = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == user.id,
        RecipeFeedback.cooked == True
    ).all()
    cooked_ids = {f.recipe_id for f in cooked_feedback}

    # -- Try KG-based recommendations --
    kg = get_kg_engine()
    if kg and user.ailments:
        recipes = await _kg_recommendations(kg, user, count, skipped_ids, cooked_ids, restrictions)
        if recipes:
            return recipes

    # -- Fallback: TheMealDB + mock --
    recipes = await fetch_recipes_from_api(count * 2)
    if len(recipes) < count:
        recipes = get_mock_recipes(count * 2)

    recipes = [r for r in recipes if r["id"] not in skipped_ids]

    # Filter out recipes containing ingredients the user should avoid
    if user.ailments:
        condition_names = {a.name for a in user.ailments}
        all_ingredients = set()
        for r in recipes:
            all_ingredients.update(r.get("ingredients", []))
        if all_ingredients:
            result = get_avoid_ingredients(condition_names, all_ingredients)
            avoid_set = {ing.lower() for ing in result.get("unified", set())}
            if avoid_set:
                recipes = [
                    r for r in recipes
                    if not any(ing.lower() in avoid_set for ing in r.get("ingredients", []))
                ]

    # Rank by nutrient-distance if user has biometric data, else random
    targets = _compute_user_targets(user)
    if targets and user.ailments:
        user_needs = set()
        for ailment in user.ailments:
            if ailment.needs:
                user_needs.update(n.strip().lower() for n in ailment.needs.split(","))
        recipes.sort(key=lambda r: _nutrient_relevance_score(r, targets, user_needs))
    else:
        random.shuffle(recipes)

    for recipe in recipes:
        recipe["previously_cooked"] = recipe["id"] in cooked_ids
        recipe["restrictions_applied"] = restrictions
        if targets:
            recipe["nutrient_targets"] = targets

    return recipes[:count]


async def _kg_recommendations(
    kg,
    user: User,
    count: int,
    skipped_ids: set,
    cooked_ids: set,
    restrictions: List[str],
) -> List[dict]:
    """
    Use the KG inference engine to get ailment-based recommendations.
    For each user ailment, runs recommend_for_ailment and merges results.
    Tries to enrich recipe names with TheMealDB data.
    """
    # Aggregate recommendations across all user ailments
    recipe_scores = {}  # recipe_name -> (best_score, nutrients)

    for ailment in user.ailments:
        ailment_name = ailment.name.lower().replace(" ", "_")
        try:
            results = kg.recommend_for_ailment(ailment_name, top_k=count * 2)
            for recipe_name, score, nutrients in results:
                if recipe_name not in recipe_scores or score < recipe_scores[recipe_name][0]:
                    recipe_scores[recipe_name] = (score, nutrients)
        except Exception as e:
            print(f"[recommender] KG recommend failed for '{ailment_name}': {e}")

    if not recipe_scores:
        return []

    # Sort by score (lower is better in RotatE distance)
    sorted_recipes = sorted(recipe_scores.items(), key=lambda x: x[1][0])

    # Build recipe dicts, try to enrich from TheMealDB
    recipes = []
    for recipe_name, (score, nutrients) in sorted_recipes:
        # Use recipe name as a stable ID for the KG recipes
        recipe_id = f"kg_{recipe_name.replace(' ', '_').lower()}"

        if recipe_id in skipped_ids:
            continue

        # Try TheMealDB lookup for image/instructions
        mealdb_data = await search_mealdb_by_name(recipe_name)

        if mealdb_data:
            recipe = mealdb_data
            recipe["kg_score"] = round(score, 4)
            recipe["kg_nutrients"] = nutrients
        else:
            # Return with just KG data
            recipe = {
                "id": recipe_id,
                "name": recipe_name,
                "image": f"https://via.placeholder.com/300x200?text={recipe_name.replace(' ', '+')}",
                "category": "KG Recommended",
                "instructions": f"Recipe recommended based on nutritional profile. Provides: {', '.join(nutrients)}.",
                "ingredients": [],
                "kg_score": round(score, 4),
                "kg_nutrients": nutrients,
            }

        recipe["previously_cooked"] = recipe.get("id", recipe_id) in cooked_ids
        recipe["restrictions_applied"] = restrictions
        recipes.append(recipe)

    # Filter with avoid_ingredients (Step 1 of pipeline — before ranking)
    condition_names = {a.name for a in user.ailments}
    all_ingredients = set()
    for r in recipes:
        all_ingredients.update(r.get("ingredients", []))
    if all_ingredients:
        result = get_avoid_ingredients(condition_names, all_ingredients)
        avoid_set = {ing.lower() for ing in result.get("unified", set())}
        if avoid_set:
            recipes = [
                r for r in recipes
                if not any(ing.lower() in avoid_set for ing in r.get("ingredients", []))
            ]

    # Re-rank with nutrient-distance scoring (Step 5 of pipeline)
    targets = _compute_user_targets(user)
    if targets:
        user_needs = set()
        for ailment in user.ailments:
            if ailment.needs:
                user_needs.update(n.strip().lower() for n in ailment.needs.split(","))
        # Combined score: KG score (RotatE distance) + nutrient relevance
        recipes.sort(key=lambda r: (
            r.get("kg_score", 0) + _nutrient_relevance_score(r, targets, user_needs)
        ))
        for recipe in recipes:
            recipe["nutrient_targets"] = targets

    return recipes[:count]


# ============ Feedback Helpers (unchanged) ============

def save_recipe_feedback(
    db: Session,
    user_id: int,
    recipe_id: str,
    recipe_name: str,
    recipe_image: str,
    recipe_data: dict,
    cooked: bool = False,
    skipped: bool = False,
    rating: Optional[int] = None
) -> RecipeFeedback:
    """Save or update user feedback for a recipe"""

    # Check if feedback already exists
    existing = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == user_id,
        RecipeFeedback.recipe_id == recipe_id
    ).first()

    if existing:
        existing.cooked = cooked
        existing.skipped = skipped
        if rating is not None:
            existing.rating = rating
        db.commit()
        db.refresh(existing)
        return existing

    # Create new feedback
    feedback = RecipeFeedback(
        user_id=user_id,
        recipe_id=recipe_id,
        recipe_name=recipe_name,
        recipe_image=recipe_image,
        recipe_data=json.dumps(recipe_data),
        cooked=cooked,
        skipped=skipped,
        rating=rating
    )

    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_user_feedback_history(
    db: Session,
    user_id: int,
    cooked_only: bool = False,
    skipped_only: bool = False
) -> List[RecipeFeedback]:
    """Get user's recipe feedback history"""

    query = db.query(RecipeFeedback).filter(RecipeFeedback.user_id == user_id)

    if cooked_only:
        query = query.filter(RecipeFeedback.cooked == True)
    elif skipped_only:
        query = query.filter(RecipeFeedback.skipped == True)

    return query.order_by(RecipeFeedback.updated_at.desc()).all()
