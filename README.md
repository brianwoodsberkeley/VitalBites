# VitalBites

Health-conscious recipe recommendation system. Users register with health conditions, and the system recommends recipes filtered by clinically contraindicated ingredients and ranked by personalized nutrient targets.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (20 recommended; use [nvm](https://github.com/nvm-sh/nvm) to manage versions)
- **PostgreSQL 14+** running locally

## Setup

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd VitalBites
```

### 2. Create the PostgreSQL database

```bash
createdb recipes
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
DATABASE_URL=postgresql://<your-username>@localhost:5432/recipes
SECRET_KEY=dev-secret-key-change-in-production
EOF
```

Replace `<your-username>` with your PostgreSQL username (often your OS username).

### 4. Install backend dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 5. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Running the app

### Backend (terminal 1)

```bash
source venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will:
- Create all database tables on startup
- Seed the ailments table with 20 health conditions
- Serve the API at http://localhost:8000
- API docs available at http://localhost:8000/docs

### Frontend (terminal 2)

```bash
cd frontend
REACT_APP_API_URL=http://localhost:8000 npm start
```

The frontend will serve at http://localhost:3000.

**Important:** Set `REACT_APP_API_URL=http://localhost:8000` explicitly to ensure the frontend talks to your local backend. If this env var is set globally to a production URL, the frontend will call the wrong server.

## Usage

1. Go to http://localhost:3000/register
2. Create an account with email, password, and select your health conditions
3. (Optional) Enter your body metrics (height, weight, age, sex, activity level) on the second registration step — these are used to calculate personalized daily nutrient targets
4. Browse your personalized recipe recommendations on the dashboard
5. Edit your health conditions and body metrics anytime from the Profile page

## How it works

1. **Ingredient filtering** — `avoid_ingredients.py` maps your health conditions to clinically contraindicated ingredients (e.g., high-sodium foods for hypertension) and removes recipes containing them
2. **Nutrient-target ranking** — `nutrient_targets.py` computes your daily nutrient targets (kcal, protein, carbs, fat, fiber, sodium, etc.) from your biometrics using Mifflin-St Jeor BMR and condition-specific DRI adjustments. Recipes are ranked by how well their ingredients align with your targets
3. **Knowledge graph** (optional) — If a trained RotatE model is present in `backend/models/`, recipes are scored by KG embedding distance and combined with nutrient-target ranking

## Docker (alternative)

```bash
docker-compose -f infrastructure/docker-compose.yml up
```

This starts the frontend (:3000), backend (:8000), and PostgreSQL (:5432) together.

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `SECRET_KEY` | Yes | — | JWT signing secret |
| `REACT_APP_API_URL` | No | `http://localhost:8000` | Backend URL for frontend API calls |
