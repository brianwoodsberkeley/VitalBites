"""
Load recipe_nutrients.json into a PostgreSQL recipe_nutrients table.

Usage:
    python scripts/load_recipe_nutrients.py

Reads DATABASE_URL from .env in the repo root.
Creates the recipe_nutrients table if it doesn't exist, then bulk-inserts
all recipes. Existing data is cleared on each run to allow re-imports.
"""

import json
import os
import sys
from pathlib import Path

# Load .env from repo root
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(repo_root / ".env")
except ImportError:
    pass

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Add it to .env or export it.")
    sys.exit(1)

JSON_PATH = repo_root / "trained_model" / "recipe_nutrients.json"
if not JSON_PATH.exists():
    print(f"ERROR: {JSON_PATH} not found.")
    sys.exit(1)

BATCH_SIZE = 5000


def main():
    print(f"Loading {JSON_PATH} ...")
    with open(JSON_PATH) as f:
        data = json.load(f)
    print(f"  {len(data)} recipes loaded from JSON.")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recipe_nutrients (
            id SERIAL PRIMARY KEY,
            recipe_name TEXT NOT NULL,
            nutrients TEXT[] NOT NULL
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_recipe_nutrients_name ON recipe_nutrients (recipe_name);")
    conn.commit()

    # Clear existing data for clean re-import
    cur.execute("TRUNCATE recipe_nutrients RESTART IDENTITY;")
    conn.commit()

    # Bulk insert in batches
    batch = []
    total = 0
    for recipe_name, nutrients in data.items():
        batch.append((recipe_name, nutrients))
        if len(batch) >= BATCH_SIZE:
            _insert_batch(cur, batch)
            total += len(batch)
            batch = []
            print(f"  {total} / {len(data)} inserted ...")

    if batch:
        _insert_batch(cur, batch)
        total += len(batch)

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done. {total} recipes inserted into recipe_nutrients table.")


def _insert_batch(cur, batch):
    args = ",".join(
        cur.mogrify("(%s, %s)", (name, nutrients)).decode()
        for name, nutrients in batch
    )
    cur.execute(f"INSERT INTO recipe_nutrients (recipe_name, nutrients) VALUES {args}")


if __name__ == "__main__":
    main()
