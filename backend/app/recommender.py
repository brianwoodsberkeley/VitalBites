import asyncio
import random
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
import httpx
from sqlalchemy.orm import Session
from .models import User, RecipeFeedback, RecipeNutrient

# Food.com recipe dataset (local parquet). Override with VITALBITES_FOODCOM_PATH.
_DEFAULT_FOODCOM_PATH = str(Path(__file__).resolve().parent.parent.parent / "df_foodcom_recipes_final.parquet")
FOODCOM_PARQUET_PATH = os.environ.get("VITALBITES_FOODCOM_PATH", _DEFAULT_FOODCOM_PATH)

# TheMealDB is used as a best-effort enrichment source for images, instructions,
# and youtube links — the Food.com parquet lacks those fields.
MEALDB_API_BASE = "https://www.themealdb.com/api/json/v1/1"

# ============ KG Model Singleton ============

# Add repo root to sys.path so we can import train_and_infer, avoid_ingredients, nutrient_targets
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/
_REPO_ROOT = _BACKEND_DIR.parent                       # VitalBites/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from avoid_ingredients import get_avoid_ingredients
from nutrient_targets import compute_nutrient_targets

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
        print(f"[recommender] Falling back to Food.com/mock recommendations.")
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


# ============ Food.com Parquet Dataset ============

_foodcom_df = None
_foodcom_name_index: dict = {}
_foodcom_load_attempted = False


def _load_foodcom_dataset():
    """Lazy-load the Food.com parquet once and build a lowercased name index."""
    global _foodcom_df, _foodcom_name_index, _foodcom_load_attempted
    if _foodcom_load_attempted:
        return _foodcom_df
    _foodcom_load_attempted = True

    if not os.path.exists(FOODCOM_PARQUET_PATH):
        print(f"[recommender] Food.com parquet not found at {FOODCOM_PARQUET_PATH}; using mock fallback.")
        return None
    try:
        import pandas as pd
        print(f"[recommender] Loading Food.com parquet from {FOODCOM_PARQUET_PATH}...")
        df = pd.read_parquet(FOODCOM_PARQUET_PATH)
        df = df[df["Name"].notna()].reset_index(drop=True)
        _foodcom_df = df
        names_lower = df["Name"].astype(str).str.lower()
        _foodcom_name_index = {}
        for idx, n in enumerate(names_lower):
            if n not in _foodcom_name_index:
                _foodcom_name_index[n] = idx
        print(f"[recommender] Loaded {len(df)} Food.com recipes.")
    except Exception as e:
        print(f"[recommender] Failed to load Food.com parquet: {e}")
        _foodcom_df = None
    return _foodcom_df


def _json_safe(value):
    """Convert a pandas/numpy value into a JSON-serialisable Python primitive."""
    import math
    if value is None:
        return None
    # Unwrap numpy scalars first (they look like floats but aren't).
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    # numpy array or pandas Series → recurse into list.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return [_json_safe(v) for v in value.tolist()]
        except Exception:
            pass
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def _format_foodcom_row(row) -> dict:
    """Convert a Food.com parquet row into the recipe dict shape the API returns."""
    ingredients_raw = row.get("RecipeIngredientParts")
    if ingredients_raw is None:
        ingredients = []
    else:
        ingredients = [str(i) for i in list(ingredients_raw) if i is not None and str(i).strip()]

    quantities_raw = row.get("RecipeIngredientQuantities")
    quantities = []
    if quantities_raw is not None:
        quantities = [str(q) for q in list(quantities_raw)]

    name = str(row.get("Name") or "").strip()
    category = row.get("RecipeCategory")
    category_str = str(category) if category is not None and str(category) != "nan" else "Food.com"

    instructions_raw = row.get("RecipeInstructions")
    instructions_steps: list = []
    if instructions_raw is not None:
        try:
            instructions_steps = [str(s).strip() for s in list(instructions_raw) if s is not None and str(s).strip()]
        except TypeError:
            pass
    instructions = "\n".join(instructions_steps)

    images_raw = row.get("Images")
    image_url = ""
    if images_raw is not None:
        try:
            for img in list(images_raw):
                if img is None:
                    continue
                s = str(img).strip()
                if s and s.lower() != "character(0)":
                    image_url = s
                    break
        except TypeError:
            pass
    if not image_url:
        image_url = f"https://via.placeholder.com/300x200?text={name.replace(' ', '+')[:60]}"

    # Structured per-serving nutrition pulled straight from the parquet.
    nutrition: dict = {}
    for parquet_col, key in [
        ("Calories_per_serving",          "calories"),
        ("ProteinContent_per_serving",    "protein_g"),
        ("CarbohydrateContent_per_serving","carbs_g"),
        ("FatContent_per_serving",        "fat_g"),
        ("SaturatedFatContent_per_serving","saturated_fat_g"),
        ("FiberContent_per_serving",      "fiber_g"),
        ("SugarContent_per_serving",      "sugar_g"),
        ("SodiumContent_per_serving",     "sodium_mg"),
        ("CholesterolContent_per_serving","cholesterol_mg"),
    ]:
        v = row.get(parquet_col)
        if v is not None and str(v) != "nan":
            try:
                nutrition[key] = float(v)
            except (TypeError, ValueError):
                pass

    # Free-text health function strings — lowercased and concatenated for
    # substring matching against user 'needs' (iron, vitamin_c, etc.).
    hf_parts = []
    for col in ("HealthFunctions", "MicronutrientHealthFunctions"):
        arr = row.get(col)
        if arr is None:
            continue
        try:
            for s in list(arr):
                if s:
                    hf_parts.append(str(s).lower())
        except TypeError:
            pass
    health_functions_text = " ".join(hf_parts)

    # Full parquet row, JSON-safe, so every column (even ones not otherwise
    # used) is visible on the client for debugging / future features.
    parquet_row: dict = {}
    try:
        for col in row.index:
            parquet_row[str(col)] = _json_safe(row[col])
    except Exception as e:
        print(f"[recommender] parquet_row dump failed for '{name}': {e}")

    recipe_id = f"foodcom_{int(row.name)}"

    return {
        "id": recipe_id,
        "name": name,
        "image": image_url,
        "category": category_str,
        "area": "Food.com",
        "instructions": instructions,
        "ingredients": ingredients,
        "ingredient_quantities": quantities,
        "source": "https://www.food.com/",
        "youtube": "",
        "nutrition": nutrition,
        "health_functions_text": health_functions_text,
        "parquet_row": parquet_row,
    }


async def search_foodcom_by_name(name: str) -> Optional[dict]:
    """Look up a recipe in the Food.com parquet by (case-insensitive) name."""
    df = _load_foodcom_dataset()
    if df is None or not name:
        return None
    idx = _foodcom_name_index.get(name.lower())
    if idx is None:
        return None
    try:
        row = df.iloc[idx]
        return _format_foodcom_row(row)
    except Exception as e:
        print(f"[recommender] Food.com lookup failed for '{name}': {e}")
        return None


# ============ TheMealDB enrichment ============

async def _fetch_mealdb_enrichment(client: httpx.AsyncClient, name: str) -> Optional[dict]:
    """Query TheMealDB for image/instructions/youtube data for a given recipe name."""
    if not name:
        return None
    try:
        r = await client.get(f"{MEALDB_API_BASE}/search.php", params={"s": name})
        if r.status_code == 200:
            data = r.json()
            if data.get("meals"):
                m = data["meals"][0]
                return {
                    "image": m.get("strMealThumb") or "",
                    "instructions": m.get("strInstructions") or "",
                    "youtube": m.get("strYoutube") or "",
                    "area": m.get("strArea") or "",
                    "source": m.get("strSource") or "",
                }
    except Exception as e:
        print(f"[recommender] MealDB enrich failed for '{name}': {e}")
    return None


def _overlay_enrichment(recipe: dict, extra: Optional[dict]) -> dict:
    """Fill empty recipe fields from an enrichment dict. Parquet data wins."""
    if not extra:
        return recipe
    for k, v in extra.items():
        if not v:
            continue
        existing = recipe.get(k)
        if existing and (not isinstance(existing, str) or "placeholder.com" not in existing):
            continue
        recipe[k] = v
    return recipe


async def _enrich_recipes_with_mealdb(recipes: List[dict]) -> List[dict]:
    """Best-effort parallel enrichment of Food.com recipes with TheMealDB fields."""
    if not recipes:
        return recipes
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            results = await asyncio.gather(
                *[_fetch_mealdb_enrichment(client, r.get("name", "")) for r in recipes],
                return_exceptions=True,
            )
        for recipe, extra in zip(recipes, results):
            if isinstance(extra, Exception):
                continue
            _overlay_enrichment(recipe, extra)
    except Exception as e:
        print(f"[recommender] MealDB enrichment batch failed: {e}")
    return recipes


async def search_mealdb_by_name(name: str) -> Optional[dict]:
    """Look up a recipe by name: prefer Food.com, enrich with TheMealDB."""
    recipe = await search_foodcom_by_name(name)
    if recipe is None:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            extra = await _fetch_mealdb_enrichment(client, name)
        _overlay_enrichment(recipe, extra)
    except Exception as e:
        print(f"[recommender] MealDB single-enrich failed for '{name}': {e}")
    return recipe


# ============ Fallback: Food.com random sample + mock ============

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
    """Sample random recipes from Food.com parquet and enrich via TheMealDB."""
    df = _load_foodcom_dataset()
    if df is None or len(df) == 0:
        return []
    try:
        sample = df.sample(n=min(count, len(df)))
        recipes = [_format_foodcom_row(sample.iloc[i]) for i in range(len(sample))]
    except Exception as e:
        print(f"[recommender] Food.com sampling failed: {e}")
        return []
    return await _enrich_recipes_with_mealdb(recipes)


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


def _bulk_lookup_nutrients(db: Session, recipe_names: List[str]) -> dict:
    """Look up nutrients for recipes from the recipe_nutrients table.
    Returns a dict of recipe_name (lowercase) -> set of nutrient names."""
    if not recipe_names:
        return {}
    rows = db.query(RecipeNutrient.recipe_name, RecipeNutrient.nutrients).filter(
        RecipeNutrient.recipe_name.in_(recipe_names)
    ).all()
    return {row.recipe_name.lower(): set(row.nutrients) for row in rows}


def _get_recipe_nutrients(recipe: dict, db_nutrients: dict) -> set:
    """Get nutrients for a recipe: prefer DB lookup, fall back to ingredient map."""
    name = (recipe.get("name") or "").lower()
    if name in db_nutrients:
        return db_nutrients[name]
    return _recipe_ingredient_nutrients(recipe)


# Assume one recipe serving ≈ 1/3 of daily intake when comparing to targets.
PER_MEAL_FRACTION = 1.0 / 3.0

# Ailment 'needs' tokens (from seed_data.py) → parquet nutrition keys.
NEED_TO_NUTRITION_KEY = {
    "protein":       "protein_g",
    "fiber":         "fiber_g",
    "carbohydrate":  "carbs_g",
    "fat":           "fat_g",
    "saturated_fat": "saturated_fat_g",
}

# Nutrients the user should not exceed. Maps to a default daily cap used
# when compute_nutrient_targets does not supply one.
CAP_KEYS_DEFAULTS = {
    "sodium_mg":       2300.0,
    "saturated_fat_g": 20.0,
    "sugar_g":         50.0,
    "cholesterol_mg":  300.0,
}

# Recipes that exceed per_meal_cap * HARD_CAP_MULTIPLIER are dropped entirely.
HARD_CAP_MULTIPLIER = 2.0


def _need_to_tokens(need: str) -> set:
    """Expand a need like 'vitamin_b9_folate' into searchable substrings."""
    tokens = {need.replace("_", " ")}
    for piece in need.split("_"):
        if len(piece) > 2 and piece != "vitamin":
            tokens.add(piece)
    return tokens


def _per_meal_cap(targets: dict, key: str) -> Optional[float]:
    """Convert a daily target into a per-serving cap."""
    daily = None
    if targets:
        daily = targets.get(key)
    if daily is None:
        daily = CAP_KEYS_DEFAULTS.get(key)
    if daily is None:
        return None
    return daily * PER_MEAL_FRACTION


def _exceeds_hard_cap(recipe: dict, targets: dict) -> bool:
    """True if any cap nutrient exceeds HARD_CAP_MULTIPLIER * per-meal share."""
    nutrition = recipe.get("nutrition") or {}
    if not nutrition:
        return False
    for key in CAP_KEYS_DEFAULTS:
        val = nutrition.get(key)
        if val is None:
            continue
        per_meal = _per_meal_cap(targets, key)
        if per_meal is None:
            continue
        if val > per_meal * HARD_CAP_MULTIPLIER:
            return True
    return False


def _nutrient_relevance_score(recipe: dict, targets: dict, user_needs: set,
                               db_nutrients: dict = None) -> float:
    """
    Rank a recipe by how well its measured per-serving nutrition aligns with
    the user's daily targets and ailment-derived needs. Lower = better.

    Prefers structured nutrition from the Food.com parquet (recipe['nutrition']).
    Falls back to ingredient-substring heuristics only when structured data
    is unavailable.
    """
    nutrition = recipe.get("nutrition") or {}

    if nutrition:
        score = 0.0

        # Reward macros the user needs, proportional to the fraction of
        # the daily target a single serving supplies (capped at half).
        for need in user_needs:
            key = NEED_TO_NUTRITION_KEY.get(need)
            if not key:
                continue
            serving_val = nutrition.get(key)
            target_val = targets.get(key) if targets else None
            if serving_val is None or not target_val:
                continue
            frac = min(serving_val / target_val, 0.5)
            score -= frac * NUTRIENT_WEIGHTS.get(key, 1.0) * 2.0

        # Penalise cap overages — how badly serving exceeds its per-meal share.
        for cap_key in CAP_KEYS_DEFAULTS:
            serving_val = nutrition.get(cap_key)
            if serving_val is None:
                continue
            per_meal = _per_meal_cap(targets, cap_key)
            if per_meal is None or per_meal <= 0:
                continue
            if serving_val > per_meal:
                excess = (serving_val - per_meal) / per_meal
                score += excess * NUTRIENT_WEIGHTS.get(cap_key, 1.0)

        # Micronutrient / vitamin needs via HealthFunctions prose matching.
        hf_text = recipe.get("health_functions_text") or ""
        if hf_text:
            for need in user_needs:
                if need in NEED_TO_NUTRITION_KEY:
                    continue
                for tok in _need_to_tokens(need):
                    if tok in hf_text:
                        score -= 0.5
                        break

        return score

    # -- Fallback: ingredient-substring heuristic (legacy behaviour) --
    if db_nutrients is not None:
        recipe_nutrients = _get_recipe_nutrients(recipe, db_nutrients)
    else:
        recipe_nutrients = _recipe_ingredient_nutrients(recipe)

    score = 0.0
    for nutrient_tag in recipe_nutrients:
        if nutrient_tag in user_needs:
            score -= 1.0

    cap_nutrients = {
        "sodium":        ("sodium_mg",       2300.0),
        "saturated fat": ("saturated_fat_g", None),
        "saturated_fat": ("saturated_fat_g", None),
        "sugar":         ("sugar_g",         None),
        "cholesterol":   ("cholesterol_mg",  300.0),
    }
    for nutrient_tag, (target_key, default_cap) in cap_nutrients.items():
        if nutrient_tag in recipe_nutrients and targets:
            target_val = targets.get(target_key, default_cap)
            if target_val is not None and default_cap is not None:
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
        recipes = await _kg_recommendations(kg, user, count, skipped_ids, cooked_ids, restrictions, db)
        if recipes:
            return recipes

    # -- Fallback: Food.com parquet + mock --
    # Oversample heavily because avoid-ingredient + hard-cap filters
    # can remove a large fraction of random samples.
    recipes = await fetch_recipes_from_api(count * 8)
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

    targets = _compute_user_targets(user)

    # Hard-cap filter: drop recipes whose structured nutrition blows through
    # any cap by more than HARD_CAP_MULTIPLIER × per-meal share.
    if user.ailments:
        before = len(recipes)
        recipes = [r for r in recipes if not _exceeds_hard_cap(r, targets)]
        if before and before > len(recipes):
            print(f"[recommender] hard-cap filter dropped {before - len(recipes)}/{before}")

    # Rank by nutrient-distance if user has biometric data, else random
    if targets and user.ailments:
        user_needs = set()
        for ailment in user.ailments:
            if ailment.needs:
                user_needs.update(n.strip().lower() for n in ailment.needs.split(","))
        recipe_names = [r.get("name", "") for r in recipes]
        db_nutrients = _bulk_lookup_nutrients(db, recipe_names)
        recipes.sort(key=lambda r: _nutrient_relevance_score(r, targets, user_needs, db_nutrients))
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
    db: Session = None,
) -> List[dict]:
    """
    Use the KG inference engine to get ailment-based recommendations.
    For each user ailment, runs recommend_for_ailment and merges results.
    Resolves recipe names via the Food.com parquet and enriches with TheMealDB
    for image/instructions/youtube data when available.
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

    # Build recipe dicts from Food.com, enrich with TheMealDB when possible
    recipes = []
    for recipe_name, (score, nutrients) in sorted_recipes:
        # Use recipe name as a stable ID for the KG recipes
        recipe_id = f"kg_{recipe_name.replace(' ', '_').lower()}"

        if recipe_id in skipped_ids:
            continue

        # Resolve via Food.com parquet + TheMealDB enrichment (image/instructions/youtube)
        enriched = await search_mealdb_by_name(recipe_name)

        if enriched:
            recipe = enriched
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

    targets = _compute_user_targets(user)

    # Hard-cap filter — drop recipes that blow through cap nutrients by >2×.
    before = len(recipes)
    recipes = [r for r in recipes if not _exceeds_hard_cap(r, targets)]
    if before and before > len(recipes):
        print(f"[recommender] KG hard-cap filter dropped {before - len(recipes)}/{before}")

    # Re-rank with nutrient-distance scoring (Step 5 of pipeline)
    if targets:
        user_needs = set()
        for ailment in user.ailments:
            if ailment.needs:
                user_needs.update(n.strip().lower() for n in ailment.needs.split(","))
        # Bulk-lookup nutrients from DB for all candidate recipes
        db_nutrients = {}
        if db:
            recipe_names = [r.get("name", "") for r in recipes]
            db_nutrients = _bulk_lookup_nutrients(db, recipe_names)
        # Combined score: KG score (RotatE distance) + nutrient relevance
        recipes.sort(key=lambda r: (
            r.get("kg_score", 0) + _nutrient_relevance_score(r, targets, user_needs, db_nutrients)
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
