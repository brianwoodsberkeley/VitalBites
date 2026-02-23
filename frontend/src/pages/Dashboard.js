import React, { useState, useEffect } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { getCurrentUser, getRecommendations, getRecommendationsByUser, submitFeedback, getFeedbackHistory, getFeedbackHistoryByUser, logout, isLoggedIn, deleteFeedback } from '../services/api';
import { ALL_AILMENTS } from '../data/ailments';
import Logo from '../components/Logo';
import Throbber from '../components/Throbber';
import '../styles.css';

function Dashboard() {
  const navigate = useNavigate();
  const { userId: urlUserId } = useParams();
  const [user, setUser] = useState(null);
  const [recipes, setRecipes] = useState([]);
  const [cookedHistory, setCookedHistory] = useState([]);
  const [skippedHistory, setSkippedHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [recipesLoading, setRecipesLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('recommendations');
  const [menuOpen, setMenuOpen] = useState(false);
  const [throbberSeed, setThrobberSeed] = useState(0);

  const isOwner = isLoggedIn() && user && String(localStorage.getItem('userId')) === String(user.id);
  const effectiveUserId = urlUserId || localStorage.getItem('userId');

  const cacheKey = `dashboard_${effectiveUserId}`;

  useEffect(() => {
    // Try to restore from session cache first
    const cached = sessionStorage.getItem(cacheKey);
    if (cached) {
      try {
        const data = JSON.parse(cached);
        setUser(data.user);
        setRecipes(data.recipes);
        setCookedHistory(data.cookedHistory);
        setSkippedHistory(data.skippedHistory);
        setLoading(false);
        setRecipesLoading(false);
        return;
      } catch (e) {
        sessionStorage.removeItem(cacheKey);
      }
    }
    loadData();
  }, [urlUserId]);

  const saveToCache = (userData, recipesData, cooked, skipped) => {
    try {
      sessionStorage.setItem(cacheKey, JSON.stringify({
        user: userData,
        recipes: recipesData,
        cookedHistory: cooked,
        skippedHistory: skipped,
      }));
    } catch (e) { /* ignore quota errors */ }
  };

  const loadData = async () => {
    try {
      setLoading(true);
      setRecipesLoading(true);
      setThrobberSeed(s => s + 1);

      let userData;
      if (urlUserId) {
        const { getUserById } = await import('../services/api');
        userData = await getUserById(urlUserId);
      } else {
        userData = await getCurrentUser();
      }
      setUser(userData);
      setLoading(false);

      const uid = urlUserId || userData.id;

      const [recsData, cooked, skipped] = await Promise.all([
        urlUserId ? getRecommendationsByUser(uid) : getRecommendations(),
        urlUserId ? getFeedbackHistoryByUser(uid, true, false) : getFeedbackHistory(true, false),
        urlUserId ? getFeedbackHistoryByUser(uid, false, true) : getFeedbackHistory(false, true),
      ]);

      const recipesData = recsData.recipes || recsData;
      setRecipes(recipesData);
      setCookedHistory(cooked);
      setSkippedHistory(skipped);
      saveToCache(userData, recipesData, cooked, skipped);
    } catch (err) {
      console.error('Failed to load data:', err);
      if (!urlUserId) {
        navigate('/login');
      }
    } finally {
      setLoading(false);
      setRecipesLoading(false);
    }
  };

  const handleRefresh = async () => {
    sessionStorage.removeItem(cacheKey);
    setRefreshing(true);
    setThrobberSeed(s => s + 1);
    try {
      const recsData = urlUserId
        ? await getRecommendationsByUser(effectiveUserId)
        : await getRecommendations();
      setRecipes(recsData.recipes || recsData);
    } catch (err) {
      console.error('Failed to refresh:', err);
    } finally {
      setRefreshing(false);
    }
  };

  const handleCooked = async (recipe, e) => {
    e.stopPropagation();
    try {
      await submitFeedback(recipe, true, false);
      setRecipes(prev => prev.filter(r => r.id !== recipe.id));
      const cooked = await getFeedbackHistory(true, false);
      setCookedHistory(cooked);
      sessionStorage.removeItem(cacheKey);
    } catch (err) {
      console.error('Failed to mark as cooked:', err);
    }
  };

  const handleSkip = async (recipe, e) => {
    e.stopPropagation();
    try {
      await submitFeedback(recipe, false, true);
      setRecipes(prev => prev.filter(r => r.id !== recipe.id));
      const skipped = await getFeedbackHistory(false, true);
      setSkippedHistory(skipped);
      sessionStorage.removeItem(cacheKey);
    } catch (err) {
      console.error('Failed to skip:', err);
    }
  };

  const handleUndo = async (recipeId) => {
    try {
      await deleteFeedback(recipeId);
      const [cooked, skipped] = await Promise.all([
        getFeedbackHistory(true, false),
        getFeedbackHistory(false, true),
      ]);
      setCookedHistory(cooked);
      setSkippedHistory(skipped);
      sessionStorage.removeItem(cacheKey);
    } catch (err) {
      console.error('Failed to undo:', err);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleRecipeClick = (recipe) => {
    navigate(`/u/${effectiveUserId}/recipe/${recipe.id}`, { state: { recipe } });
  };

  // Get the user's ailment objects for nutrient/condition mapping
  const getUserAilments = () => {
    if (!user) return [];
    if (user.ailments && Array.isArray(user.ailments)) {
      return user.ailments;
    }
    if (user.ailment_ids && Array.isArray(user.ailment_ids)) {
      return user.ailment_ids.map(id => ALL_AILMENTS.find(a => a.id === id)).filter(Boolean);
    }
    return [];
  };

  // For a recipe, figure out which conditions it helps and what nutrients it provides
  const getRecipeHealthInfo = (recipe) => {
    // If KG data is available, use it directly
    if (recipe.kg_nutrients && recipe.kg_nutrients.length > 0) {
      const nutrients = recipe.kg_nutrients;
      const ailments = getUserAilments();
      const conditions = ailments
        .filter(a => {
          const needs = a.needs ? a.needs.split(',').map(n => n.trim().toLowerCase()) : [];
          return nutrients.some(n => needs.includes(n.toLowerCase()));
        })
        .map(a => a.name);
      return { nutrients, conditions };
    }

    // Fallback: show the user's conditions and their needed nutrients
    const ailments = getUserAilments();
    const allNeeds = new Set();
    ailments.forEach(a => {
      if (a.needs) {
        a.needs.split(',').forEach(n => allNeeds.add(n.trim()));
      }
    });
    return {
      nutrients: Array.from(allNeeds),
      conditions: ailments.map(a => a.name),
    };
  };

  const getUserAilmentNames = () => {
    if (!user) return [];
    if (user.ailments && Array.isArray(user.ailments)) {
      return user.ailments.map(a => a.name);
    }
    if (user.ailment_ids && Array.isArray(user.ailment_ids)) {
      return user.ailment_ids.map(id => {
        const ailment = ALL_AILMENTS.find(a => a.id === id);
        return ailment ? ailment.name : `Condition #${id}`;
      });
    }
    return [];
  };

  if (loading) {
    return (
      <div className="dashboard bg-dashboard">
        <div className="navbar">
          <Link to="/" className="navbar-brand"><Logo height={32} /></Link>
          <div className="navbar-user">
            <div className="skeleton skeleton-text" style={{ width: 120 }} />
            <div className="skeleton skeleton-circle" style={{ width: 40, height: 40 }} />
          </div>
        </div>
        <div className="dashboard-content">
          {/* Stats skeleton */}
          <div className="stats-bar">
            <div className="stat-item"><div className="skeleton skeleton-text" style={{ width: 40, height: 28, margin: '0 auto 0.25rem' }} /><div className="skeleton skeleton-text" style={{ width: 60, height: 12, margin: '0 auto' }} /></div>
            <div className="stat-item"><div className="skeleton skeleton-text" style={{ width: 40, height: 28, margin: '0 auto 0.25rem' }} /><div className="skeleton skeleton-text" style={{ width: 60, height: 12, margin: '0 auto' }} /></div>
            <div className="stat-item"><div className="skeleton skeleton-text" style={{ width: 40, height: 28, margin: '0 auto 0.25rem' }} /><div className="skeleton skeleton-text" style={{ width: 60, height: 12, margin: '0 auto' }} /></div>
          </div>

          {/* Health profile skeleton */}
          <div className="welcome-card">
            <div className="skeleton skeleton-text" style={{ width: 120, height: 16, marginBottom: '0.75rem' }} />
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
              <div className="skeleton skeleton-pill" /><div className="skeleton skeleton-pill" /><div className="skeleton skeleton-pill" /><div className="skeleton skeleton-pill" />
            </div>
          </div>

          {/* Tabs skeleton */}
          <div className="tabs">
            <div className="tab active">Recommendations</div>
            <div className="tab">Cooked</div>
            <div className="tab">Skipped</div>
          </div>

          {/* Recipes loading with spinner */}
          <div className="recipes-section">
            <div className="section-header">
              <div className="skeleton skeleton-text" style={{ width: 180, height: 20 }} />
              <div className="skeleton skeleton-btn" />
            </div>
            <div className="loading-spinner-container">
              <Throbber seed={throbberSeed} />
              <p className="loading-spinner-text">Loading your recipes...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard bg-dashboard">
      <div className="navbar">
        <Link to="/" className="navbar-brand"><Logo height={32} /></Link>
        <div className="navbar-user">
          {user && <span className="user-email">{user.email}</span>}
          <a
            href="https://github.com/brianwoodsberkeley/VitalBites/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug"
            target="_blank"
            rel="noopener noreferrer"
            className="bug-btn"
            title="Report a bug"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M4.355.522a.5.5 0 01.623.333l.291.956A5 5 0 018 1c1.007 0 1.946.298 2.731.811l.29-.956a.5.5 0 11.957.29l-.41 1.352A5 5 0 0113 4h.5a.5.5 0 010 1H13a5 5 0 01-.034.5H14.5a.5.5 0 010 1h-1.757a5.5 5.5 0 01-9.486 0H1.5a.5.5 0 010-1h1.534A5 5 0 013 5h-.5a.5.5 0 010-1H3a5 5 0 011.432-2.455l-.41-1.352a.5.5 0 01.333-.623zM8 2a4 4 0 00-4 4 4.5 4.5 0 008.945.5H12a4 4 0 00-4-4.5V2zM4.5 5.5a.5.5 0 000 1h7a.5.5 0 000-1h-7z"/>
            </svg>
            Report Bug
          </a>
          <a
            href="https://github.com/brianwoodsberkeley/VitalBites/issues?q=is%3Aissue%20state%3Aopen%20label%3Aenhancement"
            target="_blank"
            rel="noopener noreferrer"
            className="feature-btn"
            title="Request a feature"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 0a.5.5 0 01.5.5v2a.5.5 0 01-1 0v-2A.5.5 0 018 0zM3.146 2.646a.5.5 0 01.708 0l1.414 1.414a.5.5 0 11-.708.708L3.146 3.354a.5.5 0 010-.708zM11.44 4.06a.5.5 0 010-.708l1.414-1.414a.5.5 0 01.708.708L12.148 4.06a.5.5 0 01-.708 0zM8 5a3 3 0 100 6 3 3 0 000-6zM0 8a.5.5 0 01.5-.5h2a.5.5 0 010 1h-2A.5.5 0 010 8zm13 0a.5.5 0 01.5-.5h2a.5.5 0 010 1h-2A.5.5 0 0113 8zM4.768 11.94a.5.5 0 010 .708l-1.414 1.414a.5.5 0 11-.708-.708l1.414-1.414a.5.5 0 01.708 0zm6.464 0a.5.5 0 01.708 0l1.414 1.414a.5.5 0 01-.708.708l-1.414-1.414a.5.5 0 010-.708zM8 13a.5.5 0 01.5.5v2a.5.5 0 01-1 0v-2A.5.5 0 018 13z"/>
            </svg>
            Request Feature
          </a>
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
                  <div className="hamburger-menu-header">
                    <div className="hamburger-menu-email">{user?.email}</div>
                  </div>
                  {isOwner && (
                    <button
                      className="hamburger-menu-item"
                      onClick={() => { setMenuOpen(false); navigate(`/u/${effectiveUserId}/profile`); }}
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
        {/* Stats */}
        <div className="stats-bar">
          <div className="stat-item">
            <span className="stat-number">{recipes.length}</span>
            <span className="stat-label">Recipes</span>
          </div>
          <div className="stat-item">
            <span className="stat-number">{cookedHistory.length}</span>
            <span className="stat-label">Cooked</span>
          </div>
          <div className="stat-item">
            <span className="stat-number">{skippedHistory.length}</span>
            <span className="stat-label">Skipped</span>
          </div>
        </div>

        {/* Health Profile */}
        <div className="welcome-card">
          <div className="welcome-title">Health Profile</div>
          <div className="ailments-list">
            {getUserAilmentNames().map((name, i) => (
              <span key={i} className="ailment-tag">{name}</span>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div className="tabs">
          <button className={`tab ${activeTab === 'recommendations' ? 'active' : ''}`} onClick={() => setActiveTab('recommendations')}>
            Recommendations
          </button>
          <button className={`tab ${activeTab === 'cooked' ? 'active' : ''}`} onClick={() => setActiveTab('cooked')}>
            Cooked
          </button>
          <button className={`tab ${activeTab === 'skipped' ? 'active' : ''}`} onClick={() => setActiveTab('skipped')}>
            Skipped
          </button>
        </div>

        {/* Recommendations Tab */}
        {activeTab === 'recommendations' && (
          <div className="recipes-section">
            <div className="section-header">
              <h2>Recommended for You</h2>
              <button className="refresh-btn" onClick={handleRefresh} disabled={refreshing}>
                {refreshing ? 'Loading...' : 'Get New Recipes'}
              </button>
            </div>
            <p className="recipe-instructions-hint">Click on a recipe to see details and cooking instructions</p>
            {(recipesLoading || refreshing) ? (
              <div className="loading-spinner-container">
                <Throbber seed={throbberSeed} />
                <p className="loading-spinner-text">Loading your recipes...</p>
              </div>
            ) : recipes.length === 0 ? (
              <div className="empty-message">No recipes found. Try refreshing!</div>
            ) : (
              <div className="recipe-table-container">
                <table className="recipe-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Recipe</th>
                      <th>Category</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recipes.map((recipe, index) => {
                      const healthInfo = getRecipeHealthInfo(recipe);
                      return (
                        <tr key={recipe.id} className="recipe-table-row" onClick={() => handleRecipeClick(recipe)}>
                          <td className="recipe-table-num">{index + 1}</td>
                          <td className="recipe-table-name">
                            {recipe.name}
                            {cookedHistory.some(h => h.recipe_id === recipe.id) && (
                              <span className="cooked-badge-inline">Cooked</span>
                            )}
                          </td>
                          <td className="recipe-table-category">
                            {recipe.category || recipe.strCategory || ''}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Cooked Tab */}
        {activeTab === 'cooked' && (
          <div className="history-section">
            <h2>Cooked Recipes</h2>
            <p className="section-description">Recipes you've prepared</p>
            {cookedHistory.length === 0 ? (
              <div className="empty-message">No cooked recipes yet. Start cooking!</div>
            ) : (
              <div className="history-list">
                {cookedHistory.map((item) => (
                  <div key={item.id} className="history-item">
                    {item.recipe_image && (
                      <img src={item.recipe_image} alt={item.recipe_name} className="history-image" />
                    )}
                    <div className="history-info">
                      <h4>{item.recipe_name}</h4>
                      <p className="history-date">
                        {new Date(item.created_at).toLocaleDateString()}
                        {isOwner && (
                          <> &middot; <span className="link" onClick={() => handleUndo(item.recipe_id)} style={{ cursor: 'pointer' }}>Undo</span></>
                        )}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Skipped Tab */}
        {activeTab === 'skipped' && (
          <div className="history-section">
            <h2>Skipped Recipes</h2>
            <p className="section-description">Recipes you've passed on</p>
            {skippedHistory.length === 0 ? (
              <div className="empty-message">No skipped recipes.</div>
            ) : (
              <div className="history-list">
                {skippedHistory.map((item) => (
                  <div key={item.id} className="history-item">
                    {item.recipe_image && (
                      <img src={item.recipe_image} alt={item.recipe_name} className="history-image" />
                    )}
                    <div className="history-info">
                      <h4>{item.recipe_name}</h4>
                      <p className="history-date">
                        {new Date(item.created_at).toLocaleDateString()}
                        {isOwner && (
                          <> &middot; <span className="link" onClick={() => handleUndo(item.recipe_id)} style={{ cursor: 'pointer' }}>Undo</span></>
                        )}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
