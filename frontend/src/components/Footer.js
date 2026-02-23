import React from 'react';
import { Link } from 'react-router-dom';
import Logo from './Logo';
import NavButtons from './NavButtons';

function Footer() {
  return (
    <footer className="landing-footer">
      <div className="landing-container landing-footer-inner">
        <div className="landing-footer-brand">
          <Logo height={28} />
        </div>
        <div className="landing-footer-links">
          <Link to="/login" className="landing-footer-link">Sign In</Link>
          <Link to="/register" className="landing-footer-link">Get Started</Link>
          <NavButtons />
        </div>
        <p className="landing-footer-copy">&copy; 2026 VitalFoods AI</p>
      </div>
    </footer>
  );
}

export default Footer;
