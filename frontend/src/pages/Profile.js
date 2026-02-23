import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getCurrentUser, updateAilments, isLoggedIn } from '../services/api';
import { AILMENTS_BY_CATEGORY } from '../data/ailments';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
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

  const backTo = `/u/${userId}/dashboard`;

  useEffect(() => {
    loadUser();
  }, [userId]);

  const loadUser = async () => {
    try {
      setLoading(true);
      const userData = await getCurrentUser();
      setUser(userData);
      const ids = userData.ailments
        ? userData.ailments.map(a => a.id)
        : (userData.ailment_ids || []);
      setSelectedAilments(ids);
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

  const menuItems = [
    {
      label: 'Dashboard',
      icon: Navbar.ICON_DASHBOARD,
      onClick: () => navigate(backTo),
    },
  ];

  if (loading) {
    return (
      <div className="dashboard bg-profile">
        <Navbar backTo={backTo} />
        <div className="dashboard-content">
          <div className="loading-recipes">Loading profile...</div>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="dashboard bg-profile">
      <Navbar
        backTo={backTo}
        email={user?.email}
        menuItems={menuItems}
      />

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
      <Footer />
    </div>
  );
}

export default Profile;
