import httpx
import random
import json
from typing import List, Optional
from sqlalchemy.orm import Session
from .models import User, RecipeFeedback

# TheMealDB API - free, no API key required
MEALDB_API_BASE = "https://www.themealdb.com/api/json/v1/1"

# Fallback mock recipes if API fails
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
        "instructions": "Sauté onions and garlic. Add lentils and broth. Simmer until tender.",
        "ingredients": ["Red lentils", "Onion", "Garlic", "Carrots", "Vegetable broth", "Cumin"]
    },
    {
        "id": "mock_6",
        "name": "Baked Cod with Herbs",
        "image": "https://via.placeholder.com/300x200?text=Baked+Cod",
        "category": "Seafood",
        "instructions": "Season cod with herbs and lemon. Bake at 400°F for 15-20 minutes.",
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
        "instructions": "Sauté vegetables in curry paste. Add coconut milk and simmer.",
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
        "instructions": "Spiralize zucchini. Sauté shrimp with garlic. Toss together with olive oil.",
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


async def get_recommendations(
    user: User,
    db: Session,
    count: int = 10
) -> List[dict]:
    """
    Mock recommender that fetches recipes.
    In production, this would use the ML model to filter/rank based on:
    - User's health conditions
    - Previously skipped recipes
    - User preferences
    """
    
    # Get user's restrictions
    restrictions = []
    for ailment in user.ailments:
        restrictions.extend(ailment.dietary_restrictions.split(','))
    restrictions = list(set(restrictions))
    
    # Get previously skipped recipe IDs
    skipped_feedback = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == user.id,
        RecipeFeedback.skipped == True
    ).all()
    skipped_ids = {f.recipe_id for f in skipped_feedback}
    
    # Get previously cooked recipe IDs (for reference)
    cooked_feedback = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == user.id,
        RecipeFeedback.cooked == True
    ).all()
    cooked_ids = {f.recipe_id for f in cooked_feedback}
    
    # Try to fetch from API first
    recipes = await fetch_recipes_from_api(count + 5)  # Fetch extra to filter
    
    # Fall back to mock recipes if API fails
    if len(recipes) < count:
        recipes = get_mock_recipes(count + 5)
    
    # Filter out skipped recipes
    recipes = [r for r in recipes if r["id"] not in skipped_ids]
    
    # In production, here we would:
    # 1. Filter recipes based on dietary restrictions
    # 2. Use ML model to rank remaining recipes
    # 3. Apply diversity rules
    
    # For now, just shuffle and return
    random.shuffle(recipes)
    
    # Add metadata for the frontend
    for recipe in recipes:
        recipe["previously_cooked"] = recipe["id"] in cooked_ids
        recipe["restrictions_applied"] = restrictions
    
    return recipes[:count]


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
