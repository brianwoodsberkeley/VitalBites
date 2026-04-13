import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { getRecipeYouTube, getRecipeById, submitFeedback, isLoggedIn, getCurrentUser } from '../services/api';
import { ALL_AILMENTS } from '../data/ailments';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import '../styles.css';

function RecipeDetail() {
  const { userId, recipeId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const [recipe, setRecipe] = useState(location.state?.recipe || null);
  const [loading, setLoading] = useState(!location.state?.recipe);
  const [youtubeUrl, setYoutubeUrl] = useState(null);
  const [youtubeTitle, setYoutubeTitle] = useState(null);
  const [toast, setToast] = useState(null);
  const [userAilments, setUserAilments] = useState([]);

  const isOwner = isLoggedIn() && String(localStorage.getItem('userId')) === String(userId);
  const backTo = `/u/${userId}/dashboard`;

  const menuItems = isOwner ? [
    {
      label: 'Edit Profile',
      icon: Navbar.ICON_PROFILE,
      onClick: () => navigate(`/u/${userId}/profile`),
    },
  ] : [];

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        navigate(backTo);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate, backTo]);

  // Fetch recipe by ID if not passed via navigation state
  useEffect(() => {
    if (!recipe && recipeId) {
      setLoading(true);
      getRecipeById(recipeId)
        .then(data => {
          setRecipe(data);
          setLoading(false);
        })
        .catch(() => {
          setLoading(false);
        });
    }
  }, [recipe, recipeId]);

  // Set page title & debug log
  useEffect(() => {
    if (recipe) {
      console.log('[RecipeDetail] Full recipe record:', recipe);
      const recipeName = recipe.name || recipe.strMeal || 'Recipe';
      document.title = `${recipeName} — VitalFoods AI`;
    }
    return () => { document.title = 'VitalFoods AI'; };
  }, [recipe]);

  // YouTube link resolution — prefer a direct video URL, fall back to a
  // YouTube search query so the button always has a working destination.
  useEffect(() => {
    if (!recipe) return;
    const name = recipe.name || recipe.strMeal;

    if (name) {
      setYoutubeUrl(`https://www.youtube.com/results?search_query=${encodeURIComponent(name + ' recipe')}`);
      setYoutubeTitle('Watch on YouTube');
    }

    const existingYoutube = recipe.strYoutube || recipe.youtube;
    if (existingYoutube) {
      setYoutubeUrl(existingYoutube);
      setYoutubeTitle('Watch on YouTube');
    }

    if (name) {
      getRecipeYouTube(name)
        .then(data => {
          if (data.youtube_url) {
            setYoutubeUrl(data.youtube_url);
            setYoutubeTitle(data.title || 'Watch on YouTube');
          }
        })
        .catch(() => {});
    }
  }, [recipe]);

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
      navigate(backTo);
    } catch (err) {
      console.error('Failed to mark as cooked:', err);
    }
  };

  const handleSkip = async () => {
    try {
      await submitFeedback(recipe, false, true);
      navigate(backTo);
    } catch (err) {
      console.error('Failed to skip:', err);
    }
  };

  if (loading) {
    return (
      <div className="dashboard bg-recipe">
        <Navbar backTo={backTo} />
        <div className="dashboard-content">
          <div className="loading-recipes">Loading recipe...</div>
        </div>
        <Footer />
      </div>
    );
  }

  if (!recipe) {
    return (
      <div className="dashboard bg-recipe">
        <Navbar backTo={backTo} />
        <div className="dashboard-content">
          <div className="empty-message">Recipe not found.</div>
        </div>
        <Footer />
      </div>
    );
  }

  // Parse ingredients from recipe, deduplicating. Prefer the Food.com parquet
  // row when available so we can pair each ingredient with its quantity.
  const parquetParts = recipe.parquet_row?.RecipeIngredientParts;
  const parquetQtys = recipe.parquet_row?.RecipeIngredientQuantities;
  const ingredientsRaw = [];
  if (Array.isArray(parquetParts) && parquetParts.length > 0) {
    parquetParts.forEach((part, i) => {
      if (!part) return;
      const qty = Array.isArray(parquetQtys) ? parquetQtys[i] : null;
      const qtyStr = qty !== null && qty !== undefined && String(qty).trim() !== '' ? `${String(qty).trim()} ` : '';
      ingredientsRaw.push(`${qtyStr}${String(part).trim()}`);
    });
  } else if (recipe.ingredients && Array.isArray(recipe.ingredients)) {
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

  const instructionsRaw = recipe.instructions || recipe.strInstructions || '';
  const instructions = instructionsRaw
    .replace(/[\u25A1\u25A2\u25FB\u25FC\u25FD\u25FE\u2610\u2B1C\u2B1B\uFFFD]/g, '')
    .replace(/\n\s*\n/g, '\n')
    .trim();
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

  // Per-serving nutrition from Food.com parquet (attached server-side).
  const nutritionRows = [
    { key: 'calories',        label: 'Calories',      unit: 'kcal', digits: 0 },
    { key: 'protein_g',       label: 'Protein',       unit: 'g',    digits: 1 },
    { key: 'carbs_g',         label: 'Carbohydrates', unit: 'g',    digits: 1 },
    { key: 'fiber_g',         label: 'Fiber',         unit: 'g',    digits: 1 },
    { key: 'sugar_g',         label: 'Sugar',         unit: 'g',    digits: 1 },
    { key: 'fat_g',           label: 'Fat',           unit: 'g',    digits: 1 },
    { key: 'saturated_fat_g', label: 'Saturated Fat', unit: 'g',    digits: 1 },
    { key: 'cholesterol_mg',  label: 'Cholesterol',   unit: 'mg',   digits: 0 },
    { key: 'sodium_mg',       label: 'Sodium',        unit: 'mg',   digits: 0 },
  ];
  const nutrition = recipe.nutrition || {};
  const visibleNutrition = nutritionRows.filter(
    r => nutrition[r.key] !== undefined && nutrition[r.key] !== null
  );

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
      <Navbar
        backTo={backTo}
        email={null}
        menuItems={menuItems}
      />

      <div className="dashboard-content">
        <div className="recipe-detail-card">
          <div className="recipe-detail-title-row">
            <h1 className="recipe-detail-title">{recipeName}</h1>
            <button
              className="share-btn"
              title="Copy link to share this recipe"
              onClick={() => {
                navigator.clipboard.writeText(window.location.href).then(() => {
                  showToast('Link copied! Share it with friends.');
                });
              }}
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M11 2.5a2.5 2.5 0 11-.603 1.628l-4.823 2.412a2.5 2.5 0 110 2.92l4.823 2.412a2.5 2.5 0 11.448.894l-4.823-2.412a2.5 2.5 0 110-4.708l4.823-2.412A2.5 2.5 0 0111 2.5z"/>
              </svg>
              Share
            </button>
          </div>
          <p className="recipe-detail-meta">
            {category}{category && area ? ' · ' : ''}{area}
          </p>

          {image && (
            <img
              src={image}
              alt={recipeName}
              className="recipe-detail-image"
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
          )}

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

          {visibleNutrition.length > 0 && (
            <>
              <h3 className="recipe-detail-heading">Nutrition (per serving)</h3>
              <table className="ingredients-table">
                <thead>
                  <tr>
                    <th>Nutrient</th>
                    <th>Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleNutrition.map(r => (
                    <tr key={r.key}>
                      <td>{r.label}</td>
                      <td>{Number(nutrition[r.key]).toFixed(r.digits)} {r.unit}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
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
          <h3 className="recipe-detail-heading" style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem', marginTop: '1.5rem' }}>Buy Recipe Now</h3>
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
      <Footer />
    </div>
  );
}

export default RecipeDetail;
