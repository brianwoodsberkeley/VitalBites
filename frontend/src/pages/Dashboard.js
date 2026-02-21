import React, { useState, useEffect } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { getCurrentUser, getRecommendations, getRecommendationsByUser, submitFeedback, getFeedbackHistory, getFeedbackHistoryByUser, logout, isLoggedIn, deleteFeedback } from '../services/api';
import { ALL_AILMENTS } from '../data/ailments';
import Logo from '../components/Logo';
import '../styles.css';

function Dashboard() {
  const navigate = useNavigate();
  const { userId: urlUserId } = useParams();
  const [user, setUser] = useState(null);
  const [recipes, setRecipes] = useState([]);
  const [cookedHistory, setCookedHistory] = useState([]);
  const [skippedHistory, setSkippedHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('recommendations');
  const [menuOpen, setMenuOpen] = useState(false);

  const isOwner = isLoggedIn() && user && String(localStorage.getItem('userId')) === String(user.id);
  const effectiveUserId = urlUserId || localStorage.getItem('userId');

  useEffect(() => {
    loadData();
  }, [urlUserId]);

  const loadData = async () => {
    try {
      setLoading(true);

      let userData;
      if (urlUserId) {
        const { getUserById } = await import('../services/api');
        userData = await getUserById(urlUserId);
      } else {
        userData = await getCurrentUser();
      }
      setUser(userData);

      const uid = urlUserId || userData.id;

      const [recsData, cooked, skipped] = await Promise.all([
        urlUserId ? getRecommendationsByUser(uid) : getRecommendations(),
        urlUserId ? getFeedbackHistoryByUser(uid, true, false) : getFeedbackHistory(true, false),
        urlUserId ? getFeedbackHistoryByUser(uid, false, true) : getFeedbackHistory(false, true),
      ]);

      setRecipes(recsData.recipes || recsData);
      setCookedHistory(cooked);
      setSkippedHistory(skipped);
    } catch (err) {
      console.error('Failed to load data:', err);
      if (!urlUserId) {
        navigate('/login');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
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

  const getUserAilmentNames = () => {
    if (!user || !user.ailment_ids) return [];
    return user.ailment_ids.map(id => {
      const ailment = ALL_AILMENTS.find(a => a.id === id);
      return ailment ? ailment.name : `Condition #${id}`;
    });
  };

  if (loading) {
    return (
      <div className="dashboard bg-dashboard">
        <div className="navbar">
          <Link to="/" className="navbar-brand"><Logo height={32} /></Link>
        </div>
        <div className="dashboard-content">
          <div className="loading-recipes">Loading your dashboard...</div>
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
            {recipes.length === 0 ? (
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
                    {recipes.map((recipe, index) => (
                      <tr key={recipe.id} className="recipe-table-row" onClick={() => handleRecipeClick(recipe)}>
                        <td className="recipe-table-num">{index + 1}</td>
                        <td className="recipe-table-name">
                          {recipe.name}
                          {cookedHistory.some(h => h.recipe_id === recipe.id) && (
                            <span className="cooked-badge-inline">Cooked</span>
                          )}
                        </td>
                        <td className="recipe-table-category">{recipe.category || recipe.strCategory || '—'}</td>
                      </tr>
                    ))}
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
