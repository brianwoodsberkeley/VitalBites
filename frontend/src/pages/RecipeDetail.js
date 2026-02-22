import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation, Link } from 'react-router-dom';
import { getRecipeYouTube, submitFeedback, isLoggedIn, logout, getCurrentUser } from '../services/api';
import { ALL_AILMENTS } from '../data/ailments';
import Logo from '../components/Logo';
import '../styles.css';

function RecipeDetail() {
  const { userId, recipeId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const recipe = location.state?.recipe;

  const [youtubeUrl, setYoutubeUrl] = useState(null);
  const [youtubeTitle, setYoutubeTitle] = useState(null);
  const [toast, setToast] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [userAilments, setUserAilments] = useState([]);

  const isOwner = isLoggedIn() && String(localStorage.getItem('userId')) === String(userId);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        navigate(`/u/${userId}/dashboard`);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate, userId]);

  useEffect(() => {
    if (!recipe) {
      navigate(`/u/${userId}/dashboard`);
      return;
    }

    // Check if recipe already has a YouTube link
    const existingYoutube = recipe.strYoutube || recipe.youtube;
    if (existingYoutube) {
      setYoutubeUrl(existingYoutube);
      setYoutubeTitle('Watch on YouTube');
    }

    // Always also try our API for a better/fallback result
    getRecipeYouTube(recipe.name || recipe.strMeal)
      .then(data => {
        if (data.youtube_url) {
          setYoutubeUrl(data.youtube_url);
          setYoutubeTitle(data.title || 'Watch on YouTube');
        }
      })
      .catch(() => {});
  }, [recipe, userId, navigate]);

  useEffect(() => {
    if (isLoggedIn()) {
      getCurrentUser()
        .then(userData => {
          const ailments = userData.ailments && Array.isArray(userData.ailments)
            ? userData.ailments
            : (userData.ailment_ids || []).map(id => ALL_AILMENTS.find(a => a.id === id)).filter(Boolean);
          setUserAilments(ailments);
        })
        .catch(() => {});
    }
  }, []);

  const showToast = (message) => {
    setToast(message);
    setTimeout(() => setToast(null), 2500);
  };

  const handleCooked = async () => {
    try {
      await submitFeedback(recipe, true, false);
      navigate(`/u/${userId}/dashboard`);
    } catch (err) {
      console.error('Failed to mark as cooked:', err);
    }
  };

  const handleSkip = async () => {
    try {
      await submitFeedback(recipe, false, true);
      navigate(`/u/${userId}/dashboard`);
    } catch (err) {
      console.error('Failed to skip:', err);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  if (!recipe) return null;

  // Parse ingredients from recipe, deduplicating
  const ingredientsRaw = [];
  if (recipe.ingredients && Array.isArray(recipe.ingredients)) {
    ingredientsRaw.push(...recipe.ingredients);
  } else {
    // TheMealDB format: strIngredient1..20, strMeasure1..20
    for (let i = 1; i <= 20; i++) {
      const ing = recipe[`strIngredient${i}`];
      const measure = recipe[`strMeasure${i}`];
      if (ing && ing.trim()) {
        ingredientsRaw.push(`${measure ? measure.trim() + ' ' : ''}${ing.trim()}`);
      }
    }
  }
  const seen = new Set();
  const ingredients = ingredientsRaw.filter(item => {
    const key = item.toLowerCase().trim();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const instructions = recipe.instructions || recipe.strInstructions || '';
  const image = recipe.image || recipe.strMealThumb || '';
  const recipeName = recipe.name || recipe.strMeal || 'Recipe';
  const category = recipe.category || recipe.strCategory || '';
  const area = recipe.area || recipe.strArea || '';

  // Map nutrients from KG data to conditions the recipe helps with
  const getHelpsWithConditions = () => {
    const nutrients = recipe.kg_nutrients || [];
    if (nutrients.length > 0) {
      return userAilments
        .filter(a => {
          const needs = a.needs ? a.needs.split(',').map(n => n.trim().toLowerCase()) : [];
          return nutrients.some(n => needs.includes(n.toLowerCase()));
        })
        .map(a => a.name);
    }
    return userAilments.map(a => a.name);
  };

  const helpsWithConditions = getHelpsWithConditions();

  // Map common ingredients to their micro-nutrients
  const INGREDIENT_NUTRIENT_MAP = {
    'spinach': ['iron', 'folate', 'vitamin K1', 'magnesium', 'vitamin C'],
    'kale': ['vitamin K1', 'vitamin C', 'calcium', 'iron'],
    'broccoli': ['vitamin C', 'vitamin K1', 'folate', 'fiber', 'chromium'],
    'sweet potato': ['vitamin A', 'fiber', 'potassium', 'manganese'],
    'carrot': ['vitamin A', 'fiber', 'potassium'],
    'tomato': ['vitamin C', 'potassium', 'folate', 'vitamin K1'],
    'bell pepper': ['vitamin C', 'vitamin A', 'folate'],
    'pepper': ['vitamin C'],
    'garlic': ['selenium', 'manganese', 'vitamin B6'],
    'onion': ['vitamin C', 'chromium', 'fiber'],
    'ginger': ['magnesium', 'manganese', 'potassium'],
    'lemon': ['vitamin C', 'folate', 'potassium'],
    'lime': ['vitamin C', 'folate'],
    'orange': ['vitamin C', 'folate', 'potassium', 'fiber'],
    'banana': ['potassium', 'magnesium', 'vitamin B6', 'fiber'],
    'avocado': ['potassium', 'magnesium', 'folate', 'monounsaturated fat', 'fiber'],
    'blueberr': ['vitamin C', 'vitamin K1', 'manganese', 'fiber'],
    'strawberr': ['vitamin C', 'folate', 'manganese', 'fiber'],
    'apple': ['fiber', 'vitamin C', 'potassium'],
    'chicken': ['protein', 'selenium', 'vitamin B6', 'zinc', 'phosphorus'],
    'turkey': ['protein', 'selenium', 'zinc', 'phosphorus'],
    'beef': ['protein', 'iron', 'zinc', 'vitamin B12', 'selenium'],
    'lamb': ['protein', 'iron', 'zinc', 'vitamin B12'],
    'pork': ['protein', 'selenium', 'zinc', 'vitamin B12', 'phosphorus'],
    'salmon': ['protein', 'polyunsaturated fat', 'vitamin D', 'selenium', 'vitamin B12'],
    'tuna': ['protein', 'selenium', 'vitamin B12', 'polyunsaturated fat'],
    'shrimp': ['protein', 'selenium', 'vitamin B12', 'zinc', 'copper'],
    'cod': ['protein', 'selenium', 'vitamin B12', 'phosphorus'],
    'sardine': ['calcium', 'vitamin D', 'vitamin B12', 'polyunsaturated fat', 'selenium'],
    'mackerel': ['polyunsaturated fat', 'vitamin B12', 'selenium', 'vitamin D'],
    'egg': ['protein', 'selenium', 'vitamin B12', 'vitamin D', 'iron', 'zinc'],
    'milk': ['calcium', 'vitamin D', 'protein', 'phosphorus', 'vitamin B12'],
    'yogurt': ['calcium', 'protein', 'vitamin B12', 'phosphorus'],
    'cheese': ['calcium', 'protein', 'phosphorus', 'zinc', 'vitamin B12'],
    'butter': ['vitamin A', 'saturated fat'],
    'cream': ['calcium', 'vitamin A'],
    'tofu': ['protein', 'calcium', 'iron', 'magnesium'],
    'lentil': ['fiber', 'iron', 'folate', 'protein', 'magnesium', 'potassium'],
    'chickpea': ['fiber', 'protein', 'iron', 'folate', 'magnesium'],
    'black bean': ['fiber', 'protein', 'iron', 'folate', 'magnesium'],
    'kidney bean': ['fiber', 'protein', 'iron', 'folate'],
    'bean': ['fiber', 'protein', 'iron', 'folate'],
    'pea': ['fiber', 'protein', 'vitamin C', 'iron', 'folate'],
    'rice': ['carbohydrate', 'manganese', 'selenium'],
    'oat': ['fiber', 'magnesium', 'iron', 'zinc', 'manganese'],
    'quinoa': ['protein', 'fiber', 'magnesium', 'iron', 'manganese'],
    'pasta': ['carbohydrate', 'iron', 'folate'],
    'bread': ['carbohydrate', 'iron', 'folate', 'fiber'],
    'wheat': ['fiber', 'manganese', 'selenium', 'magnesium'],
    'almond': ['magnesium', 'vitamin E', 'calcium', 'fiber', 'protein'],
    'walnut': ['polyunsaturated fat', 'magnesium', 'copper', 'manganese'],
    'cashew': ['magnesium', 'zinc', 'iron', 'copper'],
    'peanut': ['protein', 'magnesium', 'folate', 'fiber'],
    'pistachio': ['protein', 'fiber', 'potassium', 'vitamin B6'],
    'sunflower seed': ['vitamin E', 'selenium', 'magnesium', 'copper'],
    'pumpkin seed': ['magnesium', 'zinc', 'iron', 'copper'],
    'flax': ['polyunsaturated fat', 'fiber', 'magnesium'],
    'chia': ['fiber', 'calcium', 'magnesium', 'polyunsaturated fat'],
    'sesame': ['calcium', 'magnesium', 'iron', 'zinc', 'copper'],
    'olive oil': ['monounsaturated fat', 'vitamin E', 'vitamin K1'],
    'coconut': ['manganese', 'copper', 'fiber'],
    'honey': ['manganese', 'potassium'],
    'dark chocolate': ['iron', 'magnesium', 'copper', 'manganese', 'fiber'],
    'cocoa': ['iron', 'magnesium', 'copper', 'manganese'],
    'mushroom': ['selenium', 'copper', 'vitamin D', 'zinc'],
    'potato': ['potassium', 'vitamin C', 'vitamin B6', 'fiber'],
    'corn': ['fiber', 'magnesium', 'potassium', 'manganese'],
    'celery': ['vitamin K1', 'potassium', 'folate'],
    'cucumber': ['vitamin K1', 'potassium'],
    'zucchini': ['vitamin C', 'potassium', 'manganese'],
    'squash': ['vitamin A', 'vitamin C', 'potassium', 'magnesium'],
    'asparagus': ['folate', 'vitamin K1', 'iron', 'fiber'],
    'cauliflower': ['vitamin C', 'vitamin K1', 'folate', 'fiber'],
    'cabbage': ['vitamin C', 'vitamin K1', 'fiber'],
    'lettuce': ['vitamin K1', 'folate', 'vitamin A'],
    'parsley': ['vitamin K1', 'vitamin C', 'iron', 'folate'],
    'cilantro': ['vitamin K1', 'vitamin A', 'vitamin C'],
    'basil': ['vitamin K1', 'iron', 'calcium'],
    'turmeric': ['iron', 'manganese', 'copper'],
    'cinnamon': ['manganese', 'calcium', 'iron'],
  };

  const getIngredientNutrients = (ingredient) => {
    const ingLower = ingredient.toLowerCase();
    const matched = new Set();
    for (const [key, nutrients] of Object.entries(INGREDIENT_NUTRIENT_MAP)) {
      if (ingLower.includes(key)) {
        nutrients.forEach(n => matched.add(n));
      }
    }
    return Array.from(matched);
  };

  return (
    <div className="dashboard bg-recipe">
      <div className="navbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button className="back-btn" onClick={() => navigate(`/u/${userId}/dashboard`)}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M10.354 3.354a.5.5 0 00-.708-.708l-5 5a.5.5 0 000 .708l5 5a.5.5 0 00.708-.708L5.707 8l4.647-4.646z"/>
            </svg>
            Back
          </button>
          <Link to="/" className="navbar-brand"><Logo height={32} /></Link>
        </div>
        <div className="navbar-user">
          <div className="hamburger-wrapper">
            <button className="hamburger-btn" onClick={() => setMenuOpen(!menuOpen)}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <rect y="3" width="20" height="2" rx="1" />
                <rect y="9" width="20" height="2" rx="1" />
                <rect y="15" width="20" height="2" rx="1" />
              </svg>
            </button>
            {menuOpen && (
              <>
                <div className="hamburger-backdrop" onClick={() => setMenuOpen(false)} />
                <div className="hamburger-menu">
                  {isOwner && (
                    <button
                      className="hamburger-menu-item"
                      onClick={() => { setMenuOpen(false); navigate(`/u/${userId}/profile`); }}
                    >
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 8a3 3 0 100-6 3 3 0 000 6zm0 1c-3.31 0-6 1.79-6 4v1h12v-1c0-2.21-2.69-4-6-4z"/></svg>
                      Edit Profile
                    </button>
                  )}
                  <div className="hamburger-menu-divider" />
                  <button
                    className="hamburger-menu-item hamburger-menu-item-danger"
                    onClick={() => { setMenuOpen(false); handleLogout(); }}
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M6 2a1 1 0 00-1 1v2a1 1 0 002 0V4h5v8H7v-1a1 1 0 00-2 0v2a1 1 0 001 1h6a1 1 0 001-1V3a1 1 0 00-1-1H6z"/><path d="M1.293 7.293a1 1 0 000 1.414l2 2a1 1 0 001.414-1.414L4.414 9H10a1 1 0 000-2H4.414l.293-.293a1 1 0 00-1.414-1.414l-2 2z"/></svg>
                    Sign Out
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="recipe-detail-card">
          {image && <img src={image} alt={recipeName} className="recipe-detail-image" />}
          <h1 className="recipe-detail-title">{recipeName}</h1>
          <p className="recipe-detail-meta">
            {category}{category && area ? ' · ' : ''}{area}
          </p>

          {/* Helps With */}
          {helpsWithConditions.length > 0 && (
            <div className="helps-with-section">
              <h3 className="recipe-detail-heading">Helps With</h3>
              <div className="helps-with-tags">
                {helpsWithConditions.map((c, i) => (
                  <span key={i} className="condition-tag">{c}</span>
                ))}
              </div>
            </div>
          )}

          {ingredients.length > 0 && (
            <>
              <h3 className="recipe-detail-heading">Ingredients</h3>
              <table className="ingredients-table">
                <thead>
                  <tr>
                    <th>Ingredient</th>
                    <th>Micro-nutrients</th>
                  </tr>
                </thead>
                <tbody>
                  {ingredients.map((ing, i) => {
                    const nutrients = getIngredientNutrients(ing);
                    return (
                      <tr key={i}>
                        <td>{ing}</td>
                        <td>
                          {nutrients.length > 0 ? (
                            <span className="ingredient-nutrients">
                              {nutrients.map((n, j) => (
                                <span key={j} className="nutrient-tag-sm">{n}</span>
                              ))}
                            </span>
                          ) : (
                            <span className="text-muted">—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </>
          )}

          {instructions && (
            <>
              <h3 className="recipe-detail-heading">Instructions</h3>
              <div className="instructions">{instructions}</div>
            </>
          )}

          {youtubeUrl && (
            <a href={youtubeUrl} target="_blank" rel="noopener noreferrer" className="youtube-link">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
                <path d="M23.498 6.186a3.016 3.016 0 00-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 00.502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 002.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 002.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
              </svg>
              {youtubeTitle || 'Watch on YouTube'}
            </a>
          )}

          {/* Action Buttons */}
          <div className="action-buttons-grid">
            <button className="action-btn action-btn-amazon" onClick={() => showToast('Amazon Fresh integration coming soon!')}>
              <img
                className="action-btn-logo"
                src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
                alt="Amazon Fresh"
                style={{ filter: 'brightness(0) invert(1)' }}
              />
            </button>
            <button className="action-btn action-btn-doordash" onClick={() => showToast('DoorDash integration coming soon!')}>
              <img
                className="action-btn-logo action-btn-logo-lg"
                src="https://upload.wikimedia.org/wikipedia/commons/6/6a/DoorDash_Logo.svg"
                alt="DoorDash"
                style={{ filter: 'brightness(0) invert(1)' }}
              />
            </button>
            <button className="action-btn action-btn-instacart" onClick={() => showToast('Instacart integration coming soon!')}>
              <img
                className="action-btn-logo"
                src="https://upload.wikimedia.org/wikipedia/commons/9/9f/Instacart_logo_and_wordmark.svg"
                alt="Instacart"
                style={{ filter: 'brightness(0) invert(1)' }}
              />
            </button>
            <button className="action-btn action-btn-walmart" onClick={() => showToast('Walmart integration coming soon!')}>
              <img
                className="action-btn-logo"
                src="https://upload.wikimedia.org/wikipedia/commons/b/b1/Walmart_logo_%282008%29.svg"
                alt="Walmart"
                style={{ filter: 'brightness(0) invert(1)' }}
              />
            </button>
            <button className="action-btn action-btn-wholefoods" onClick={() => showToast('Whole Foods integration coming soon!')}>
              <img
                className="action-btn-logo"
                src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Whole_Foods_Market_logo.svg"
                alt="Whole Foods"
                style={{ filter: 'brightness(0) invert(1)' }}
              />
            </button>
            <button className="action-btn action-btn-traderjoes" onClick={() => showToast("Trader Joe's integration coming soon!")}>
              <img
                className="action-btn-logo"
                src="https://upload.wikimedia.org/wikipedia/commons/d/d1/Trader_Joes_Logo.svg"
                alt="Trader Joe's"
                style={{ filter: 'brightness(0) invert(1)' }}
              />
            </button>
          </div>

          {/* Cook / Skip buttons for owner */}
          {isOwner && (
            <div className="modal-actions">
              <button className="btn-cooked large" onClick={handleCooked} style={{ flex: 1 }}>
                Mark as Cooked
              </button>
              <button className="btn-skip large" onClick={handleSkip} style={{ flex: 1 }}>
                Skip
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Toast notification */}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

export default RecipeDetail;
