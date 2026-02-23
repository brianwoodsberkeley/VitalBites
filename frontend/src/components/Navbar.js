import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { logout } from '../services/api';
import Logo from './Logo';
import NavButtons from './NavButtons';

const ICON_PROFILE = (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 8a3 3 0 100-6 3 3 0 000 6zm0 1c-3.31 0-6 1.79-6 4v1h12v-1c0-2.21-2.69-4-6-4z"/></svg>
);

const ICON_DASHBOARD = (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1l-7 6h3v6h3v-4h2v4h3V7h3L8 1z"/></svg>
);

function Navbar({ backTo, email, showEmail, menuItems = [] }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="navbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        {backTo && (
          <button className="back-btn" onClick={() => navigate(backTo)}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
              <path d="M10.354 3.354a.5.5 0 00-.708-.708l-5 5a.5.5 0 000 .708l5 5a.5.5 0 00.708-.708L5.707 8l4.647-4.646z"/>
            </svg>
            Back
          </button>
        )}
        <Link to="/" className="navbar-brand"><Logo height={32} /></Link>
      </div>
      <div className="navbar-user">
        {showEmail && email && <span className="user-email">{email}</span>}
        <NavButtons />
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
                {email && (
                  <div className="hamburger-menu-header">
                    <div className="hamburger-menu-email">{email}</div>
                  </div>
                )}
                {menuItems.map((item, i) => (
                  <button
                    key={i}
                    className={`hamburger-menu-item${item.danger ? ' hamburger-menu-item-danger' : ''}`}
                    onClick={() => { setMenuOpen(false); item.onClick(); }}
                  >
                    {item.icon}
                    {item.label}
                  </button>
                ))}
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
  );
}

Navbar.ICON_PROFILE = ICON_PROFILE;
Navbar.ICON_DASHBOARD = ICON_DASHBOARD;

export default Navbar;
