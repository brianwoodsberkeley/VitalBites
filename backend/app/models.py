from sqlalchemy import Column, Integer, String, Table, ForeignKey, DateTime, Boolean, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

# Many-to-many: users can have multiple ailments
user_ailments = Table(
    'user_ailments',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('ailment_id', Integer, ForeignKey('ailments.id'))
)

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    ailments = relationship('Ailment', secondary=user_ailments, back_populates='users')
    recipe_feedback = relationship('RecipeFeedback', back_populates='user')

class Ailment(Base):
    __tablename__ = 'ailments'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    category = Column(String, nullable=False)
    needs = Column(String, nullable=False)  # comma-separated nutrients to increase
    avoid = Column(String, nullable=True)   # comma-separated nutrients to limit (optional)

    users = relationship('User', secondary=user_ailments, back_populates='ailments')

class RecipeFeedback(Base):
    __tablename__ = 'recipe_feedback'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    recipe_id = Column(String, nullable=False)  # External recipe ID from API
    recipe_name = Column(String, nullable=False)
    recipe_image = Column(String, nullable=True)
    recipe_data = Column(Text, nullable=True)  # JSON string of full recipe data
    cooked = Column(Boolean, default=False)
    skipped = Column(Boolean, default=False)
    rating = Column(Integer, nullable=True)  # 1-5 rating
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship('User', back_populates='recipe_feedback')
