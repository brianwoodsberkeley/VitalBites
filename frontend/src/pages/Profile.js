import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { getCurrentUser, updateAilments, isLoggedIn, logout } from '../services/api';
import { AILMENTS_BY_CATEGORY } from '../data/ailments';
import Logo from '../components/Logo';
import '../styles.css';

function Profile() {
  const { userId } = useParams();
  const navigate = useNavigate();

  const [user, setUser] = useState(null);
  const [selectedAilments, setSelectedAilments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);

  const isOwner = isLoggedIn() && String(localStorage.getItem('userId')) === String(userId);

  useEffect(() => {
    loadUser();
  }, [userId]);

  const loadUser = async () => {
    try {
      setLoading(true);
      const userData = await getCurrentUser();
      setUser(userData);
      setSelectedAilments(userData.ailment_ids || []);
    } catch (err) {
      console.error('Failed to load user:', err);
      navigate('/login');
    } finally {
      setLoading(false);
    }
  };

  const handleAilmentToggle = (ailmentId) => {
    setSelectedAilments(prev =>
      prev.includes(ailmentId)
        ? prev.filter(id => id !== ailmentId)
        : [...prev, ailmentId]
    );
  };

  const handleSave = async () => {
    if (selectedAilments.length === 0) {
      setError('Please select at least one health condition');
      return;
    }

    setSaving(true);
    setError('');
    setSuccess('');

    try {
      await updateAilments(selectedAilments);
      setSuccess('Health profile updated successfully!');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  if (loading) {
    return (
      <div className="dashboard bg-profile">
        <div className="navbar">
          <Link to="/" className="navbar-brand"><Logo height={32} /></Link>
        </div>
        <div className="dashboard-content">
          <div className="loading-recipes">Loading profile...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard bg-profile">
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
                  <div className="hamburger-menu-header">
                    <div className="hamburger-menu-email">{user?.email}</div>
                  </div>
                  <button
                    className="hamburger-menu-item"
                    onClick={() => { setMenuOpen(false); navigate(`/u/${userId}/dashboard`); }}
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1l-7 6h3v6h3v-4h2v4h3V7h3L8 1z"/></svg>
                    Dashboard
                  </button>
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
          <h1 className="recipe-detail-title">Edit Health Profile</h1>
          <p className="recipe-detail-meta">{user?.email}</p>

          {error && <div className="error">{error}</div>}
          {success && <div className="success">{success}</div>}

          <div className="form-group">
            <label className="label">Health Conditions</label>
            <p style={{ fontSize: '0.8rem', color: '#666', marginBottom: '0.5rem' }}>
              Select all that apply to personalize your recommendations
            </p>
            <div className="ailment-selector">
              {Object.entries(AILMENTS_BY_CATEGORY).map(([category, ailments]) => (
                <div key={category} className="ailment-category">
                  <div className="category-title">{category}</div>
                  {ailments.map((ailment) => (
                    <div
                      key={ailment.id}
                      className="ailment-item"
                      onClick={() => handleAilmentToggle(ailment.id)}
                    >
                      <input
                        type="checkbox"
                        className="ailment-checkbox"
                        checked={selectedAilments.includes(ailment.id)}
                        onChange={() => handleAilmentToggle(ailment.id)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <div className="ailment-info">
                        <div className="ailment-name">{ailment.name}</div>
                        <div className="ailment-restrictions">
                          Needs: {ailment.needs}
                          {ailment.avoid && <> · Avoid: {ailment.avoid}</>}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
            {selectedAilments.length > 0 && (
              <div className="selected-count">
                {selectedAilments.length} condition{selectedAilments.length !== 1 ? 's' : ''} selected
              </div>
            )}
          </div>

          <button className="button" onClick={handleSave} disabled={saving}>
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Profile;
