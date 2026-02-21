# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VitalBites is a health-conscious recipe recommendation system. Users register with health conditions (ailments), and the system recommends recipes based on knowledge graph embeddings trained on nutritional data.

## Architecture

**Three-layer system:**

1. **ML Pipeline** (Python scripts at repo root) — Mines knowledge from recipe data, builds a knowledge graph as triples, trains RotatE embeddings via PyKEEN, and generates recipe recommendations.

2. **Backend** (`backend/`) — FastAPI app with PostgreSQL. Handles auth (JWT/OAuth2), user ailment profiles, recipe recommendations (via TheMealDB API + mock fallback), and feedback tracking.

3. **Frontend** (`frontend/`) — React 18 SPA. Login/register flow, dashboard with recipe cards, cook/skip feedback, history tabs.

### ML Pipeline Flow

```
CSV/Parquet recipes → mine_knowledge.py (Claude API) → mined_config.json
                    → csv_to_triples_vitalbites.py    → triples.tsv
                    → train_and_infer.py (RotatE)     → trained_model/
                    → create_embeddings.py (TF-IDF)   → recipe_embeddings.csv
```

Key relation types in the knowledge graph: `CONTAINS_INGREDIENT`, `IN_CATEGORY`, `HIGH_IN`, `PROVIDES`, `HAS_HEALTH_FUNCTION`, `SUBSTITUTES_FOR`, `BENEFITS_FROM`, `SHOULD_AVOID`.

### Backend Structure

- `backend/app/main.py` — FastAPI app with CORS, routes
- `backend/app/models.py` — SQLAlchemy models: User, Ailment, RecipeFeedback (user↔ailment many-to-many)
- `backend/app/auth.py` — JWT token creation/validation, password hashing
- `backend/app/routes/` — API endpoints for auth, ailments, recommendations, feedback

### Frontend Structure

- `frontend/src/App.js` — Router with ProtectedRoute wrapper
- `frontend/src/pages/` — Login, Register (with ailment selection from 42 conditions), Dashboard
- `frontend/src/services/api.js` — API client with Bearer token auth
- `frontend/src/data/ailments.js` — Ailment definitions organized by 9 categories

## Development Commands

### Frontend
```bash
cd frontend && npm start          # Dev server on :3000
cd frontend && npm run build      # Production build
```

### Backend
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Full Stack (Docker)
```bash
docker-compose -f infrastructure/docker-compose.yml up
# frontend :3000, backend :8000, postgres :5432
```

### ML Pipeline
```bash
# 1. Mine knowledge (requires Claude API key)
python mine_knowledge.py all --input df_foodcom_recipes_filtered.csv --api-key <key>

# 2. Generate triples
python csv_to_triples_vitalbites.py --input df_foodcom_recipes_filtered.csv --output triples.tsv

# 3. Train RotatE model
python train_and_infer.py train --triples triples.tsv --epochs 300 --dim 256

# 4. Generate embeddings
python create_embeddings.py

# 5. Get recommendations
python train_and_infer.py recommend --ailment "anemia" --top 10
```

## Environment Variables

- `DATABASE_URL` — PostgreSQL connection string
- `SECRET_KEY` — JWT signing secret
- `REACT_APP_API_URL` — Backend URL for frontend API calls

## Key Dependencies

- **ML**: pykeen, torch, pandas, scikit-learn, numpy
- **Backend**: fastapi, sqlalchemy, psycopg2, python-jose, passlib
- **Frontend**: react, react-router-dom

## Large Data Files (not in git)

CSV, parquet, TSV, and model files are large and untracked. The primary dataset is `df_foodcom_recipes_filtered.parquet` (32MB) / `.csv` (1.5GB) containing 312K+ FoodCom recipes.
