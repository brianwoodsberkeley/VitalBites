import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  getCurrentUser, 
  logout, 
  isLoggedIn, 
  getRecommendations, 
  submitFeedback,
  getFeedbackHistory 
} from '../services/api';
import '../styles.css';

function Dashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recipes, setRecipes] = useState([]);
  const [loadingRecipes, setLoadingRecipes] = useState(false);
  const [stats, setStats] = useState({ skipped: 0, cooked: 0 });
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [activeTab, setActiveTab] = useState('recommendations');
  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    if (!isLoggedIn()) {
      navigate('/login');
      return;
    }

    const fetchUser = async () => {
      try {
        const userData = await getCurrentUser();
        setUser(userData);
        // Load recommendations after getting user
        loadRecommendations();
      } catch (err) {
        console.error('Failed to fetch user:', err);
        logout();
        navigate('/login');
      } finally {
        setLoading(false);
      }
    };

    fetchUser();
  }, [navigate]);

  const loadRecommendations = async () => {
    setLoadingRecipes(true);
    try {
      const data = await getRecommendations(10);
      setRecipes(data.recipes);
      setStats({ skipped: data.skipped_count, cooked: data.cooked_count });
    } catch (err) {
      console.error('Failed to load recommendations:', err);
    } finally {
      setLoadingRecipes(false);
    }
  };

  const loadHistory = async (cookedOnly = false, skippedOnly = false) => {
    setLoadingHistory(true);
    try {
      const data = await getFeedbackHistory(cookedOnly, skippedOnly);
      setHistory(data);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleCooked = async (recipe) => {
    try {
      await submitFeedback(recipe, true, false);
      // Remove from current list and update stats
      setRecipes(prev => prev.filter(r => r.id !== recipe.id));
      setStats(prev => ({ ...prev, cooked: prev.cooked + 1 }));
    } catch (err) {
      console.error('Failed to mark as cooked:', err);
    }
  };

  const handleSkip = async (recipe) => {
    try {
      await submitFeedback(recipe, false, true);
      // Remove from current list and update stats
      setRecipes(prev => prev.filter(r => r.id !== recipe.id));
      setStats(prev => ({ ...prev, skipped: prev.skipped + 1 }));
    } catch (err) {
      console.error('Failed to skip recipe:', err);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'cooked') {
      loadHistory(true, false);
    } else if (tab === 'skipped') {
      loadHistory(false, true);
    }
  };

  if (loading) {
    return (
      <div className="container">
        <div className="card">
          <p className="text-center">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <nav className="navbar">
        <div className="navbar-brand">🥗 Recipe Recommender</div>
        <div className="navbar-user">
          <span className="user-email">{user?.email}</span>
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </nav>

      <div className="dashboard-content">
        {/* Stats Bar */}
        <div className="stats-bar">
          <div className="stat-item">
            <span className="stat-number">{stats.cooked}</span>
            <span className="stat-label">Recipes Cooked</span>
          </div>
          <div className="stat-item">
            <span className="stat-number">{stats.skipped}</span>
            <span className="stat-label">Recipes Skipped</span>
          </div>
          <div className="stat-item">
            <span className="stat-number">{user?.ailments?.length || 0}</span>
            <span className="stat-label">Health Conditions</span>
          </div>
        </div>

        {/* Health Conditions */}
        <div className="welcome-card">
          <h3 style={{ marginBottom: '0.75rem', color: '#333' }}>Your Health Profile</h3>
          <div className="ailments-list">
            {user?.ailments?.length > 0 ? (
              user.ailments.map((ailment) => (
                <span key={ailment.id} className="ailment-tag">
                  {ailment.name}
                </span>
              ))
            ) : (
              <p style={{ color: '#888' }}>No conditions selected</p>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'recommendations' ? 'active' : ''}`}
            onClick={() => handleTabChange('recommendations')}
          >
            Recommendations
          </button>
          <button 
            className={`tab ${activeTab === 'cooked' ? 'active' : ''}`}
            onClick={() => handleTabChange('cooked')}
          >
            Cooked
          </button>
          <button 
            className={`tab ${activeTab === 'skipped' ? 'active' : ''}`}
            onClick={() => handleTabChange('skipped')}
          >
            Skipped
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'recommendations' && (
          <div className="recipes-section">
            <div className="section-header">
              <h2>Today's Recommendations</h2>
              <button 
                className="refresh-btn" 
                onClick={loadRecommendations}
                disabled={loadingRecipes}
              >
                {loadingRecipes ? 'Loading...' : '🔄 Get New Recipes'}
              </button>
            </div>

            {loadingRecipes ? (
              <div className="loading-recipes">
                <p>Finding delicious recipes for you...</p>
              </div>
            ) : recipes.length > 0 ? (
              <div className="recipes-grid">
                {recipes.map((recipe) => (
                  <div key={recipe.id} className="recipe-card">
                    <div 
                      className="recipe-image"
                      style={{ backgroundImage: `url(${recipe.image})` }}
                      onClick={() => setSelectedRecipe(recipe)}
                    >
                      {recipe.previously_cooked && (
                        <span className="cooked-badge">✓ Cooked Before</span>
                      )}
                    </div>
                    <div className="recipe-content">
                      <h3 
                        className="recipe-name"
                        onClick={() => setSelectedRecipe(recipe)}
                      >
                        {recipe.name}
                      </h3>
                      <p className="recipe-category">{recipe.category} {recipe.area && `• ${recipe.area}`}</p>
                      <div className="recipe-actions">
                        <button 
                          className="btn-cooked"
                          onClick={() => handleCooked(recipe)}
                        >
                          ✓ I Cooked This
                        </button>
                        <button 
                          className="btn-skip"
                          onClick={() => handleSkip(recipe)}
                        >
                          ✗ Skip
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="placeholder-card">
                <div className="placeholder-icon">🍳</div>
                <h3 className="placeholder-title">No recipes available</h3>
                <p className="placeholder-text">
                  Click "Get New Recipes" to load recommendations!
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'cooked' && (
          <div className="history-section">
            <h2>Recipes You've Cooked</h2>
            {loadingHistory ? (
              <p>Loading...</p>
            ) : history.length > 0 ? (
              <div className="history-list">
                {history.map((item) => (
                  <div key={item.id} className="history-item">
                    {item.recipe_image && (
                      <img src={item.recipe_image} alt={item.recipe_name} className="history-image" />
                    )}
                    <div className="history-info">
                      <h4>{item.recipe_name}</h4>
                      <p className="history-date">
                        Cooked on {new Date(item.updated_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty-message">You haven't cooked any recipes yet!</p>
            )}
          </div>
        )}

        {activeTab === 'skipped' && (
          <div className="history-section">
            <h2>Skipped Recipes</h2>
            <p className="section-description">
              These recipes won't appear in your recommendations anymore.
            </p>
            {loadingHistory ? (
              <p>Loading...</p>
            ) : history.length > 0 ? (
              <div className="history-list">
                {history.map((item) => (
                  <div key={item.id} className="history-item">
                    {item.recipe_image && (
                      <img src={item.recipe_image} alt={item.recipe_name} className="history-image" />
                    )}
                    <div className="history-info">
                      <h4>{item.recipe_name}</h4>
                      <p className="history-date">
                        Skipped on {new Date(item.updated_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty-message">No skipped recipes.</p>
            )}
          </div>
        )}
      </div>

      {/* Recipe Detail Modal */}
      {selectedRecipe && (
        <div className="modal-overlay" onClick={() => setSelectedRecipe(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedRecipe(null)}>×</button>
            <img src={selectedRecipe.image} alt={selectedRecipe.name} className="modal-image" />
            <h2>{selectedRecipe.name}</h2>
            <p className="modal-category">{selectedRecipe.category} {selectedRecipe.area && `• ${selectedRecipe.area}`}</p>
            
            <h3>Ingredients</h3>
            <ul className="ingredients-list">
              {selectedRecipe.ingredients?.map((ing, idx) => (
                <li key={idx}>{ing}</li>
              ))}
            </ul>
            
            <h3>Instructions</h3>
            <p className="instructions">{selectedRecipe.instructions}</p>
            
            {selectedRecipe.youtube && (
              <a href={selectedRecipe.youtube} target="_blank" rel="noopener noreferrer" className="youtube-link">
                📺 Watch Video Tutorial
              </a>
            )}
            
            <div className="modal-actions">
              <button 
                className="btn-cooked large"
                onClick={() => {
                  handleCooked(selectedRecipe);
                  setSelectedRecipe(null);
                }}
              >
                ✓ I Cooked This
              </button>
              <button 
                className="btn-skip large"
                onClick={() => {
                  handleSkip(selectedRecipe);
                  setSelectedRecipe(null);
                }}
              >
                ✗ Skip This Recipe
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
