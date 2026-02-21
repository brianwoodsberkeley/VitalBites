import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation, Link } from 'react-router-dom';
import { getRecipeYouTube, submitFeedback, isLoggedIn, logout } from '../services/api';
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

  const isOwner = isLoggedIn() && String(localStorage.getItem('userId')) === String(userId);

  useEffect(() => {
    if (!recipe) {
      navigate(`/u/${userId}/dashboard`);
      return;
    }

    // Check if recipe already has a YouTube link
    if (recipe.strYoutube) {
      setYoutubeUrl(recipe.strYoutube);
      setYoutubeTitle('Watch on YouTube');
    } else {
      // Fetch from our API
      getRecipeYouTube(recipe.name || recipe.strMeal)
        .then(data => {
          if (data.youtube_url) {
            setYoutubeUrl(data.youtube_url);
            setYoutubeTitle(data.title || 'Watch on YouTube');
          }
        })
        .catch(() => {});
    }
  }, [recipe, userId, navigate]);

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

          {ingredients.length > 0 && (
            <>
              <h3 className="recipe-detail-heading">Ingredients</h3>
              <ul className="ingredients-list">
                {ingredients.map((ing, i) => (
                  <li key={i}>{ing}</li>
                ))}
              </ul>
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
