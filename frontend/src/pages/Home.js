import React from 'react';
import { Link } from 'react-router-dom';
import { AILMENTS_BY_CATEGORY } from '../data/ailments';
import Logo from '../components/Logo';
import '../styles.css';

const STEPS = [
  {
    num: 1,
    title: 'Share your health profile',
    desc: 'Tell us your conditions and dietary needs.',
  },
  {
    num: 2,
    title: 'Get matched recipes',
    desc: 'Our knowledge graph finds recipes optimized for your health.',
  },
  {
    num: 3,
    title: 'Cook, track & improve',
    desc: 'Log meals, track progress, get better recommendations over time.',
  },
];

const FEATURES = [
  {
    title: 'USDA-Backed Data',
    desc: 'Nutritional analysis grounded in real science, not influencer trends.',
    icon: '\u{1F4CA}',
  },
  {
    title: '20 Health Conditions',
    desc: 'From diabetes to anemia, personalized to your needs.',
    icon: '\u{1FA7A}',
  },
  {
    title: 'Knowledge Graph AI',
    desc: 'RotatE embeddings find hidden nutritional relationships.',
    icon: '\u{1F9E0}',
  },
  {
    title: 'Cook & Track',
    desc: 'Log what you make, skip what you don\u2019t, and watch your recommendations improve.',
    icon: '\u{1F373}',
  },
];

const CATEGORY_COLORS = {
  'Cardiovascular': '#e74c3c',
  'Metabolic / Endocrine': '#e67e22',
  'Kidney & Bone': '#f1c40f',
  'Digestive': '#2ecc71',
  'Blood & Immunity': '#1abc9c',
  'Neurological & Mental Health': '#3498db',
  'Musculoskeletal': '#9b59b6',
  'Skin & Hair': '#e84393',
  "Women's Health": '#e07a5f',
};

function Home() {
  return (
    <div className="landing-page">
      {/* Navbar */}
      <nav className="landing-nav">
        <div className="landing-nav-inner">
          <Link to="/" className="landing-brand">
            <Logo height={36} />
          </Link>
          <div className="landing-nav-actions">
            <Link to="/login" className="landing-nav-link">Sign In</Link>
            <Link to="/register" className="landing-btn-primary">Get Started</Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="landing-hero">
        <div className="landing-hero-inner">
          <Logo height={160} />
          <h1 className="landing-hero-title">Recipes that heal. Powered by science.</h1>
          <p className="landing-hero-sub">
            Tell us your health goals and we'll recommend recipes matched to your nutritional needs
            — backed by USDA data, not guesswork.
          </p>
          <div className="landing-hero-actions">
            <Link to="/register" className="landing-btn-primary landing-btn-lg">
              Get Started — It's Free
            </Link>
          </div>
          <p className="landing-hero-signin">
            Already have an account? <Link to="/login" className="link">Sign in</Link>
          </p>
        </div>
      </section>

      {/* How It Works */}
      <section className="landing-section">
        <div className="landing-container">
          <h2 className="landing-section-title">How It Works</h2>
          <div className="landing-steps">
            {STEPS.map((step) => (
              <div key={step.num} className="landing-step">
                <div className="landing-step-num">{step.num}</div>
                <h3 className="landing-step-title">{step.title}</h3>
                <p className="landing-step-desc">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="landing-section landing-section-alt">
        <div className="landing-container">
          <h2 className="landing-section-title">Why VitalBites</h2>
          <div className="landing-features">
            {FEATURES.map((f) => (
              <div key={f.title} className="landing-feature-card">
                <div className="landing-feature-icon">{f.icon}</div>
                <h3 className="landing-feature-title">{f.title}</h3>
                <p className="landing-feature-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About / Story */}
      <section className="landing-section">
        <div className="landing-container">
          <div className="landing-story">
            <h2 className="landing-story-headline">Eat for how you want to feel.</h2>
            <p className="landing-story-lede">
              Your doctor says eat better. The internet gives you a million conflicting opinions.
              VitalBites gives you actual recipes matched to your actual health needs — no guesswork,
              no fad diets, no generic meal plans.
            </p>
            <p className="landing-story-body">
              Tell us what you're dealing with — anemia, high blood pressure, diabetes, osteoporosis,
              or any of 20+ conditions — and VitalBites recommends recipes rich in the specific nutrients
              your body needs while flagging the ones to avoid. Every recommendation is backed by real
              USDA FoodData Central nutritional data, verified down to the ingredient level across
              300,000+ recipes.
            </p>

            <div className="landing-story-columns">
              <div className="landing-story-col">
                <h3 className="landing-story-col-title">How it works</h3>
                <p>
                  You select a health condition. Our AI identifies the nutrients you need (and the ones
                  to limit), then searches thousands of recipes to find the best matches — ranked by how
                  well they deliver what your body is asking for. You get a scored list of recipes you can
                  actually cook tonight, with full nutritional breakdowns showing exactly why each one was
                  recommended.
                </p>
              </div>
              <div className="landing-story-col">
                <h3 className="landing-story-col-title">What makes VitalBites different</h3>
                <p>
                  Most nutrition apps count calories. VitalBites understands the relationship between what
                  you eat and how you feel. Our knowledge graph maps the connections between ingredients,
                  nutrients, and health conditions using machine learning trained on millions of data points
                  from the USDA. The result is recommendations that a registered dietitian would
                  recognize — not advice from an algorithm that thinks a protein bar and a spinach salad
                  are the same thing.
                </p>
              </div>
            </div>

            <div className="landing-story-footer">
              <h3 className="landing-story-col-title">Built by researchers, grounded in science.</h3>
              <p>
                VitalBites was created by a team of UC Berkeley graduate students in artificial intelligence
                and data science. Every nutrient value comes from USDA FoodData Central. Every health
                condition mapping is based on clinical nutrition guidelines. We built this because we believe
                the gap between nutrition science and your dinner plate shouldn't require a PhD to close.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Supported Conditions */}
      <section className="landing-section landing-section-alt">
        <div className="landing-container">
          <h2 className="landing-section-title">Supported Health Conditions</h2>
          <p className="landing-section-sub">Personalized recommendations for 20 conditions across 9 categories</p>
          <div className="landing-conditions">
            {Object.entries(AILMENTS_BY_CATEGORY).map(([category, ailments]) => (
              <div key={category} className="landing-condition-group">
                <h4
                  className="landing-condition-cat"
                  style={{ borderLeftColor: CATEGORY_COLORS[category] || 'var(--primary)' }}
                >
                  {category}
                </h4>
                <div className="landing-condition-pills">
                  {ailments.map((a) => (
                    <span key={a.id} className="landing-condition-pill">{a.name}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="landing-container landing-footer-inner">
          <div className="landing-footer-brand">
            <Logo height={28} />
          </div>
          <div className="landing-footer-links">
            <Link to="/login" className="landing-footer-link">Sign In</Link>
            <Link to="/register" className="landing-footer-link">Get Started</Link>
          </div>
          <p className="landing-footer-copy">&copy; 2026 VitalBites</p>
        </div>
      </footer>
    </div>
  );
}

export default Home;
