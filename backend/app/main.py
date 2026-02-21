from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
from datetime import timedelta

from .database import engine, get_db, Base
from .models import User, Ailment, RecipeFeedback
from .schemas import (
    UserCreate, UserOut, AilmentOut, Token,
    RecipeFeedbackCreate, RecipeFeedbackOut, RecommendationsResponse,
    YouTubeResponse
)
from .auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from .seed_data import seed_ailments
import httpx
import os
from .recommender import (
    get_recommendations,
    save_recipe_feedback,
    get_user_feedback_history
)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Recipe Recommender API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://brianosaurus.com", "https://www.brianosaurus.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    db = next(get_db())
    seed_ailments(db)
    db.close()

# ============ Health Check ============

@app.get("/")
def root():
    return {"status": "running", "message": "Recipe Recommender API"}

@app.get("/health")
def health():
    return {"healthy": True}

# ============ Ailments ============

@app.get("/ailments", response_model=List[AilmentOut])
def get_ailments(db: Session = Depends(get_db)):
    """Get all ailments for the dropdown"""
    return db.query(Ailment).order_by(Ailment.category, Ailment.name).all()

@app.get("/ailments/categories")
def get_ailments_by_category(db: Session = Depends(get_db)):
    """Get ailments grouped by category"""
    ailments = db.query(Ailment).order_by(Ailment.category, Ailment.name).all()

    categories = {}
    for ailment in ailments:
        if ailment.category not in categories:
            categories[ailment.category] = []
        categories[ailment.category].append({
            "id": ailment.id,
            "name": ailment.name,
            "needs": ailment.needs,
            "avoid": ailment.avoid or ""
        })

    return categories

# ============ Authentication ============

@app.post("/auth/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Hash password
    hashed_password = get_password_hash(user.password)

    # Get ailments
    ailments = db.query(Ailment).filter(Ailment.id.in_(user.ailment_ids)).all()

    # Create user
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        ailments=ailments
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token"""
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"user_id": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": access_token, "token_type": "bearer"}

# ============ User Profile ============

@app.get("/users/me", response_model=UserOut)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user's profile"""
    return current_user

@app.put("/users/me/ailments", response_model=UserOut)
def update_user_ailments(
    ailment_ids: List[int],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user's ailments"""
    ailments = db.query(Ailment).filter(Ailment.id.in_(ailment_ids)).all()
    current_user.ailments = ailments
    db.commit()
    db.refresh(current_user)
    return current_user

# ============ Recommendations ============

@app.get("/recommendations", response_model=RecommendationsResponse)
async def get_recipe_recommendations(
    count: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized recipe recommendations"""

    # Get recommendations from mock recommender
    recipes = await get_recommendations(current_user, db, count)

    # Get user's avoid list
    restrictions = []
    for ailment in current_user.ailments:
        if ailment.avoid:
            restrictions.extend(ailment.avoid.split(','))
    restrictions = list(set(r for r in restrictions if r))

    # Get feedback counts
    skipped_count = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == current_user.id,
        RecipeFeedback.skipped == True
    ).count()

    cooked_count = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == current_user.id,
        RecipeFeedback.cooked == True
    ).count()

    return {
        "recipes": recipes,
        "user_restrictions": restrictions,
        "skipped_count": skipped_count,
        "cooked_count": cooked_count
    }

# ============ Recipe Feedback ============

@app.post("/feedback", response_model=RecipeFeedbackOut)
def submit_feedback(
    feedback: RecipeFeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for a recipe (cooked or skipped)"""

    result = save_recipe_feedback(
        db=db,
        user_id=current_user.id,
        recipe_id=feedback.recipe_id,
        recipe_name=feedback.recipe_name,
        recipe_image=feedback.recipe_image,
        recipe_data=feedback.recipe_data or {},
        cooked=feedback.cooked,
        skipped=feedback.skipped,
        rating=feedback.rating
    )

    return result

@app.get("/feedback/history", response_model=List[RecipeFeedbackOut])
def get_feedback_history(
    cooked_only: bool = False,
    skipped_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's recipe feedback history"""

    return get_user_feedback_history(
        db=db,
        user_id=current_user.id,
        cooked_only=cooked_only,
        skipped_only=skipped_only
    )

# ============ YouTube Search ============

@app.get("/recipe/youtube", response_model=YouTubeResponse)
async def get_recipe_youtube(name: str):
    """Search YouTube for a recipe video"""
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return {"youtube_url": None, "title": None}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "q": f'"{name}" recipe',
                    "type": "video",
                    "maxResults": 1,
                    "key": api_key,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            if items:
                video_id = items[0]["id"]["videoId"]
                title = items[0]["snippet"]["title"]
                return {
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": title,
                }
    except Exception:
        pass

    return {"youtube_url": None, "title": None}

# ============ Public User Endpoints (no auth) ============

@app.get("/users/{user_id}", response_model=UserOut)
def get_user_by_id(user_id: int, db: Session = Depends(get_db)):
    """Get a user's profile by ID (public, no auth required)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/{user_id}/recommendations", response_model=RecommendationsResponse)
async def get_user_recommendations(
    user_id: int,
    count: int = 10,
    db: Session = Depends(get_db)
):
    """Get recipe recommendations for a user by ID (public, no auth required)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    recipes = await get_recommendations(user, db, count)

    restrictions = []
    for ailment in user.ailments:
        if ailment.avoid:
            restrictions.extend(ailment.avoid.split(','))
    restrictions = list(set(r for r in restrictions if r))

    skipped_count = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == user.id,
        RecipeFeedback.skipped == True
    ).count()

    cooked_count = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == user.id,
        RecipeFeedback.cooked == True
    ).count()

    return {
        "recipes": recipes,
        "user_restrictions": restrictions,
        "skipped_count": skipped_count,
        "cooked_count": cooked_count
    }

@app.get("/users/{user_id}/feedback/history", response_model=List[RecipeFeedbackOut])
def get_user_feedback_history_by_id(
    user_id: int,
    cooked_only: bool = False,
    skipped_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get a user's feedback history by ID (public, no auth required)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return get_user_feedback_history(
        db=db,
        user_id=user.id,
        cooked_only=cooked_only,
        skipped_only=skipped_only
    )

@app.delete("/feedback/{recipe_id}")
def delete_feedback(
    recipe_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete feedback for a recipe (un-skip or un-cook)"""

    feedback = db.query(RecipeFeedback).filter(
        RecipeFeedback.user_id == current_user.id,
        RecipeFeedback.recipe_id == recipe_id
    ).first()

    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")

    db.delete(feedback)
    db.commit()

    return {"message": "Feedback deleted"}
