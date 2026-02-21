from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime

# Ailment schemas
class AilmentBase(BaseModel):
    name: str
    category: str
    needs: str
    avoid: Optional[str] = ""

class AilmentOut(AilmentBase):
    id: int

    class Config:
        from_attributes = True

# User schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=60)
    ailment_ids: List[int]

class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., max_length=60)

class UserOut(BaseModel):
    id: int
    email: str
    created_at: datetime
    ailments: List[AilmentOut]

    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[int] = None

# Recipe schemas
class RecipeOut(BaseModel):
    id: str
    name: str
    image: Optional[str] = None
    category: Optional[str] = None
    area: Optional[str] = None
    instructions: Optional[str] = None
    ingredients: List[str] = []
    source: Optional[str] = None
    youtube: Optional[str] = None
    previously_cooked: bool = False
    restrictions_applied: List[str] = []

class RecipeFeedbackCreate(BaseModel):
    recipe_id: str
    recipe_name: str
    recipe_image: Optional[str] = None
    recipe_data: Optional[dict] = None
    cooked: bool = False
    skipped: bool = False
    rating: Optional[int] = None

class RecipeFeedbackOut(BaseModel):
    id: int
    recipe_id: str
    recipe_name: str
    recipe_image: Optional[str] = None
    cooked: bool
    skipped: bool
    rating: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class RecommendationsResponse(BaseModel):
    recipes: List[RecipeOut]
    user_restrictions: List[str]
    skipped_count: int
    cooked_count: int

class YouTubeResponse(BaseModel):
    youtube_url: Optional[str] = None
    title: Optional[str] = None
