import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { login } from '../services/api';
import '../styles.css';

function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Check for success message from registration
  const successMessage = location.state?.message;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!email || !password) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);

    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">Welcome Back</h1>
        <p className="subtitle">Sign in to get your personalized recipes</p>

        {successMessage && <div className="success">{successMessage}</div>}
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
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <button type="submit" className="button" disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <p className="text-center mt-2">
          Don't have an account? <Link to="/register" className="link">Create one</Link>
        </p>
      </div>
    </div>
  );
}

export default Login;
