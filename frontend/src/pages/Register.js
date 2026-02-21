import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AILMENTS_BY_CATEGORY } from '../data/ailments';
import { register } from '../services/api';
import Logo from '../components/Logo';
import '../styles.css';

function Register() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [selectedAilments, setSelectedAilments] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAilmentToggle = (ailmentId) => {
    setSelectedAilments(prev =>
      prev.includes(ailmentId)
        ? prev.filter(id => id !== ailmentId)
        : [...prev, ailmentId]
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Validation
    if (!email || !password) {
      setError('Please fill in all required fields');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 6 || password.length > 60) {
      setError('Password must be between 6 and 60 characters');
      return;
    }

    if (selectedAilments.length === 0) {
      setError('Please select at least one health condition');
      return;
    }

    setLoading(true);

    try {
      await register(email, password, selectedAilments);
      navigate('/login', { state: { message: 'Registration successful! Please log in.' } });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container register-bg">
      <div className="brand-header">
        <Link to="/">
          <Logo height={140} />
        </Link>
      </div>
      <div className="masthead">
        <h1 className="masthead-title">Recipes that heal. Powered by science.</h1>
        <p className="masthead-subtitle">Tell us your health goals and we'll recommend recipes matched to your nutritional needs — backed by USDA data, not guesswork.</p>
      </div>
      <div className="card">
        <h1 className="title">Create Account</h1>
        <p className="subtitle">Get personalized recipe recommendations</p>

        {error && <div className="error">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="label">Email</label>
            <input
              type="email"
              className="input"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="label">Password</label>
            <input
              type="password"
              className="input"
              placeholder="At least 6 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="label">Confirm Password</label>
            <input
              type="password"
              className="input"
              placeholder="Confirm your password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
          </div>

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

          <button type="submit" className="button" disabled={loading}>
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <p className="text-center mt-2">
          Already have an account? <Link to="/login" className="link">Sign in</Link>
        </p>
      </div>
    </div>
  );
}

export default Register;
