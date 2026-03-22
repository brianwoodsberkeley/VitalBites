import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { updateProfile } from '../services/api';
import Logo from '../components/Logo';
import Footer from '../components/Footer';
import '../styles.css';

const ACTIVITY_OPTIONS = [
  { value: 'sedentary', label: 'Sedentary (little or no exercise)' },
  { value: 'light', label: 'Lightly Active (1–3 days/week)' },
  { value: 'moderate', label: 'Moderately Active (3–5 days/week)' },
  { value: 'active', label: 'Active (6–7 days/week)' },
  { value: 'very_active', label: 'Very Active (physical job or 2x/day)' },
];

function NutrientProfile() {
  const navigate = useNavigate();
  const { userId } = useParams();
  const [heightCm, setHeightCm] = useState('');
  const [weightKg, setWeightKg] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('');
  const [activityLevel, setActivityLevel] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showInterstitial, setShowInterstitial] = useState(false);

  const dashboardPath = userId ? `/u/${userId}/dashboard` : '/dashboard';

  useEffect(() => {
    if (showInterstitial) {
      const timer = setTimeout(() => navigate(dashboardPath), 10000);
      return () => clearTimeout(timer);
    }
  }, [showInterstitial, navigate, dashboardPath]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const profileData = {};
      if (heightCm) profileData.height_cm = parseFloat(heightCm);
      if (weightKg) profileData.weight_kg = parseFloat(weightKg);
      if (age) profileData.age = parseInt(age, 10);
      if (sex) profileData.sex = sex;
      if (activityLevel) profileData.activity_level = activityLevel;

      await updateProfile(profileData);
      setShowInterstitial(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (showInterstitial) {
    return (
      <div className="interstitial-overlay">
        <div className="interstitial-content">
          <div className="spinner interstitial-spinner" />
          <h2>Retrieving your recipe recommendations</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="container register-bg">
      <div className="brand-header">
        <Logo height={140} />
      </div>
      <div className="masthead">
        <h1 className="masthead-title">Tell us about yourself</h1>
        <p className="masthead-subtitle">
          We use this to calculate your personalized daily nutrient targets — calories, protein, carbs, fat, and more — based on established dietary reference intakes.
        </p>
      </div>
      <div className="card">
        <h1 className="title">Body Metrics</h1>
        <p className="subtitle">Help us personalize your nutrient targets</p>

        {error && <div className="error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="label">Height (cm)</label>
            <input
              type="number"
              className="input"
              placeholder="e.g. 170"
              min="50"
              max="300"
              step="0.1"
              value={heightCm}
              onChange={(e) => setHeightCm(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="label">Weight (kg)</label>
            <input
              type="number"
              className="input"
              placeholder="e.g. 70"
              min="20"
              max="500"
              step="0.1"
              value={weightKg}
              onChange={(e) => setWeightKg(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="label">Age</label>
            <input
              type="number"
              className="input"
              placeholder="e.g. 30"
              min="1"
              max="150"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="label">Sex</label>
            <div className="radio-group">
              <label className="radio-label">
                <input
                  type="radio"
                  name="sex"
                  value="male"
                  checked={sex === 'male'}
                  onChange={(e) => setSex(e.target.value)}
                />
                Male
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name="sex"
                  value="female"
                  checked={sex === 'female'}
                  onChange={(e) => setSex(e.target.value)}
                />
                Female
              </label>
            </div>
          </div>

          <div className="form-group">
            <label className="label">Activity Level</label>
            <select
              className="input"
              value={activityLevel}
              onChange={(e) => setActivityLevel(e.target.value)}
            >
              <option value="">Select activity level</option>
              {ACTIVITY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <button type="submit" className="button" disabled={loading}>
            {loading ? 'Saving...' : 'Save & Continue'}
          </button>
        </form>

        <p className="text-center mt-2">
          <span className="link" onClick={() => setShowInterstitial(true)} style={{ cursor: 'pointer' }}>
            Skip for now
          </span>
        </p>
      </div>
      <Footer />
    </div>
  );
}

export default NutrientProfile;
