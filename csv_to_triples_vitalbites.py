"""
VitalBites CSV → Knowledge Graph Triples Pipeline
==================================================
Converts df_foodcom_recipes_filtered.csv directly into triples.tsv
for RotatE training. No intermediate JSON needed.

Your CSV structure:
  - Name, RecipeCategory, cleaned_ingredients (numpy-style arrays)
  - Macros: Calories, FatContent, ProteinContent, FiberContent, etc.
  - Per-serving macros: Calories_per_serving, etc.
  - HealthFunctions: LONG text descriptions (not short labels)
  - MicronutrientHealthFunctions: LONG text descriptions → we extract nutrient names
  - matched_fdc_ids: USDA FoodData Central IDs

The big challenge: Your CSV has ~9.7M LINES but that's because the
HealthFunctions and MicronutrientHealthFunctions fields contain newlines
inside quoted strings. A proper CSV parser handles this, but we need to
be careful with memory for a dataset this large.

Usage:
    # Full pipeline — one command
    python csv_to_triples_vitalbites.py --input df_foodcom_recipes_filtered.csv --output triples.tsv

    # Then train RotatE
    python train_and_infer.py train --triples triples.tsv --epochs 300 --dim 256

Requirements:
    pip install pandas   (for CSV parsing with multiline fields)

Author: Brian / LeeroyChainkins AI — VitalBites Project
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
#  SECTION 1: MICRONUTRIENT DESCRIPTION → ENTITY NAME MAPPING
# ══════════════════════════════════════════════════════════════════
#
# Your MicronutrientHealthFunctions column contains long descriptions
# like "vital for oxygen delivery in blood, energy metabolism..."
# We need to map these to short entity names like "iron".
#
# This is a keyword-match approach. Each key is a UNIQUE substring
# from the description text. The value is the micronutrient name
# that will become an entity in the knowledge graph.
# ══════════════════════════════════════════════════════════════════

MICRO_DESCRIPTION_TO_NUTRIENT = {
    # Iron
    "vital for oxygen delivery in blood, energy metabolism, and immune system function": "iron",

    # Copper
    "vital for redox reactions, energy production, and cellular respiration": "copper",
    "helps with iron absorption, energy production, and the formation of collagen and neurotransmitters": "copper",

    # Vitamin C
    "functions as an antioxidant, supports collagen synthesis, aids immune cell function, enhances iron absorption, and supports neurotransmitter synthesis": "vitamin_c",

    # Folate
    "essential for dna synthesis, cell division, fetal development, and the prevention of neural tube defects": "folate",

    # Calcium
    "supports bone and teeth strength, muscle function, nerve signaling, and blood coagulation": "calcium",

    # Phosphorus
    "essential for bone mineralization, energy storage and transfer, and cell membrane integrity": "phosphorus",

    # Zinc
    "supports immune health, cellular growth, enzyme function, and wound repair": "zinc",

    # Magnesium
    "plays a role in enzymatic reactions including energy metabolism, muscle and nerve function, and dna synthesis": "magnesium",

    # Potassium
    "maintains normal heart rhythm, muscle contractions, and nerve impulses by balancing fluids and electrolytes": "potassium",

    # Sodium
    "controls blood volume and pressure, supports nerve impulse transmission and muscle contraction": "sodium",

    # Selenium
    "protects cells from oxidative damage and supports thyroid hormone metabolism": "selenium",

    # Vitamin E
    "protects membranes from oxidative stress, modulates inflammation, supports the immune system, and may have anti-cancer effects": "vitamin_e",

    # Vitamin A
    "crucial for vision, immune response, skin health, and cellular differentiation": "vitamin_a",

    # Vitamin D
    "regulates calcium and phosphorus homeostasis, supports bone health, and contributes to immune function": "vitamin_d",

    # Vitamin K1
    "activates clotting factors and is essential for blood coagulation and bone metabolism": "vitamin_k1",

    # Vitamin K2
    "cofactor for gamma-carboxylation of proteins, supports bone mineralization, vascular health, myelin repair, and has anti-inflammatory effects": "vitamin_k2",

    # Vitamin B12
    "vital for red blood cell production, neurological function, and dna synthesis": "vitamin_b12",
    "they are vital for red blood cell production, neurological function, and dna synthesis": "vitamin_b12",

    # Vitamin B6
    "coenzyme in over 140 reactions, including amino acid metabolism, neurotransmitter synthesis, hemoglobin production, and supports the immune system": "vitamin_b6",

    # Thiamin (B1)
    "cofactor for enzymes involved in carbohydrate metabolism, supports energy production, nerve function, and mitochondrial health": "thiamin",

    # Niacin (B3)
    "central role in redox reactions, energy production, dna repair, nervous system function, and maintaining the skin barrier": "niacin",

    # Pantothenic Acid (B5)
    "required for fatty acid metabolism, synthesis of coenzyme a, and energy production": "pantothenic_acid",

    # Biotin (B7)
    "important for fatty acid synthesis, gluconeogenesis, and amino acid metabolism": "biotin",

    # Manganese
    "involved in bone formation, metabolism of carbohydrates and amino acids, and antioxidant protection": "manganese",

    # Iodine
    "regulates metabolism, growth, and development via thyroid hormone synthesis": "iodine",

    # Fatty acids (general)
    "fatty acids serve as a key energy source, components of triglycerides and phospholipids": "fatty_acids",

    # Riboflavin (B2) — add if seen in your data
    "acts as a coenzyme in oxidation-reduction reactions, supports energy production": "riboflavin",
}


# ══════════════════════════════════════════════════════════════════
#  SECTION 2: HEALTH FUNCTION DESCRIPTION → SHORT LABEL MAPPING
# ══════════════════════════════════════════════════════════════════
#
# Your HealthFunctions column has descriptions like:
# "fats provide a dense source of energy, aid in the absorption..."
# We extract short, graph-friendly labels.
# ══════════════════════════════════════════════════════════════════

HEALTH_DESCRIPTION_TO_LABEL = {
    "fats provide a dense source of energy": "energy-from-fat",
    "improves bowel regularity": "digestive-health",
    "lowers cholesterol": "cholesterol-support",
    "modulates blood sugar": "blood-sugar-support",
    "supports microbiota": "gut-health",
    "proteins serve as enzymes": "protein-function",
    "muscle contraction, tissue repair": "muscle-repair",
    "controls blood volume and pressure": "blood-pressure-support",
    "carbohydrates are the body's main source of energy": "energy-from-carbs",
    "regulate blood sugar levels": "blood-sugar-support",
    "support gut health": "gut-health",
    "storing energy": "energy-storage",
    "cell communication": "cell-signaling",
}


# ══════════════════════════════════════════════════════════════════
#  SECTION 3: MACRONUTRIENT THRESHOLDS (per serving)
# ══════════════════════════════════════════════════════════════════
#
# We create HIGH_IN edges when a recipe's per-serving value
# exceeds these thresholds. Tune based on your data distribution.
# ══════════════════════════════════════════════════════════════════

MACRO_THRESHOLDS = {
    "ProteinContent_per_serving": ("protein", 15.0),      # >15g/serving = high protein
    "FiberContent_per_serving": ("fiber", 5.0),            # >5g/serving = high fiber
    "FatContent_per_serving": ("fat", 20.0),               # >20g/serving = high fat
    "SaturatedFatContent_per_serving": ("saturated_fat", 10.0),
    "SugarContent_per_serving": ("sugar", 15.0),           # >15g = high sugar
    "SodiumContent_per_serving": ("sodium_high", 400.0),   # >400mg = high sodium
}

CALORIE_THRESHOLDS = {
    "low-calorie": (0, 200),       # per serving
    "moderate-calorie": (200, 400),
    "high-calorie": (400, float("inf")),
}


# ══════════════════════════════════════════════════════════════════
#  SECTION 4: INGREDIENT SUBSTITUTIONS (curated knowledge)
# ══════════════════════════════════════════════════════════════════

SUBSTITUTIONS = {
    "ghee": ["butter", "clarified butter", "coconut oil"],
    "butter": ["ghee", "olive oil", "margarine", "coconut oil"],
    "olive oil": ["avocado oil", "vegetable oil", "canola oil"],
    "vegetable oil": ["canola oil", "sunflower oil", "corn oil", "peanut oil"],
    "heavy cream": ["coconut cream", "cashew cream", "whipping cream"],
    "sour cream": ["greek yogurt", "plain yogurt"],
    "yogurt": ["greek yogurt", "sour cream"],
    "plain yogurt": ["greek yogurt", "vanilla yogurt"],
    "soy sauce": ["tamari", "coconut aminos"],
    "parmesan cheese": ["pecorino romano", "asiago cheese"],
    "cheddar cheese": ["monterey jack", "colby cheese"],
    "feta cheese": ["goat cheese", "queso fresco"],
    "cream cheese": ["mascarpone", "neufchatel cheese"],
    "brown sugar": ["coconut sugar", "honey", "maple syrup"],
    "sugar": ["honey", "maple syrup", "agave nectar"],
    "granulated sugar": ["sugar", "honey", "coconut sugar"],
    "spaghetti": ["linguine", "fettuccine", "bucatini"],
    "rice noodles": ["glass noodles", "vermicelli"],
    "basmati rice": ["jasmine rice", "longgrain rice", "long grain rice"],
    "chicken breast": ["chicken thigh", "turkey breast", "boneless chicken"],
    "boneless chicken": ["chicken breast", "chicken thigh"],
    "chicken thigh": ["chicken breast", "turkey thigh"],
    "ground beef": ["ground turkey", "ground lamb", "ground bison"],
    "salmon": ["trout", "arctic char", "steelhead"],
    "shrimp": ["prawns", "langoustine"],
    "red lentils": ["yellow lentils", "green lentils"],
    "spinach": ["kale", "swiss chard", "collard greens"],
    "romaine lettuce": ["butter lettuce", "iceberg lettuce"],
    "milk": ["almond milk", "oat milk", "soy milk", "whole milk"],
    "heavy whipping cream": ["coconut cream", "heavy cream"],
    "lemon juice": ["lime juice", "white wine vinegar"],
    "fresh lemon juice": ["lime juice", "lemon juice"],
    "white wine": ["dry vermouth", "chicken broth"],
    "all-purpose flour": ["whole wheat flour", "bread flour"],
    "tortilla": ["corn tortilla", "flour tortilla", "naan"],
}


# ══════════════════════════════════════════════════════════════════
#  SECTION 5: AILMENT → MICRONUTRIENT MAPPINGS (curated)
# ══════════════════════════════════════════════════════════════════

AILMENT_NUTRIENT_MAP = {
    "anemia": {"needs": ["iron", "vitamin_b12", "folate", "copper", "vitamin_c"], "avoid": []},
    "osteoporosis": {"needs": ["calcium", "vitamin_d", "vitamin_k1", "vitamin_k2", "phosphorus", "magnesium"], "avoid": []},
    "hypertension": {"needs": ["potassium", "magnesium", "calcium"], "avoid": ["sodium_high"]},
    "depression": {"needs": ["fatty_acids", "vitamin_d", "folate", "vitamin_b12", "zinc", "magnesium"], "avoid": []},
    "type2_diabetes": {"needs": ["fiber", "magnesium", "manganese"], "avoid": ["sugar"]},
    "inflammation": {"needs": ["fatty_acids", "vitamin_c", "vitamin_e", "selenium"], "avoid": []},
    "fatigue": {"needs": ["iron", "vitamin_b12", "vitamin_d", "magnesium", "thiamin", "niacin"], "avoid": []},
    "weak_immunity": {"needs": ["vitamin_c", "zinc", "vitamin_d", "selenium", "vitamin_a"], "avoid": []},
    "muscle_cramps": {"needs": ["magnesium", "potassium", "calcium", "sodium"], "avoid": []},
    "poor_digestion": {"needs": ["fiber"], "avoid": []},
    "poor_bone_health": {"needs": ["calcium", "vitamin_d", "vitamin_k1", "phosphorus", "manganese"], "avoid": []},
    "poor_skin_health": {"needs": ["vitamin_a", "vitamin_c", "vitamin_e", "zinc", "biotin"], "avoid": []},
    "neural_tube_risk": {"needs": ["folate", "vitamin_b12"], "avoid": []},
    "thyroid_issues": {"needs": ["iodine", "selenium"], "avoid": []},
}


# ══════════════════════════════════════════════════════════════════
#  SECTION 6: PARSING UTILITIES
# ══════════════════════════════════════════════════════════════════

def parse_numpy_array(raw: str) -> list[str]:
    """
    Parse numpy-style array strings from your CSV.

    Handles formats like:
        ['blueberries' 'granulated sugar' 'vanilla yogurt' 'lemon juice']
        ['saffron' 'milk' 'hot green chili peppers' 'onions' 'garlic' 'clove'
         'peppercorns' 'cardamom seed' ...]
    
    Note: These span MULTIPLE LINES in the CSV which is why the file
    has 9.7M lines for ~500K recipes.
    """
    if not raw or str(raw).strip() in ("", "nan", "None", "[]", "['']"):
        return []

    raw = str(raw).strip()
    # Extract all single-quoted strings
    items = re.findall(r"'([^']*?)'", raw)
    if not items:
        # Try double-quoted
        items = re.findall(r'"([^"]*?)"', raw)
    return [normalize_ingredient(i) for i in items if i.strip()]


def normalize_ingredient(name: str) -> str:
    """Normalize ingredient name for consistent entity naming."""
    name = name.lower().strip()
    # Remove quantity prefixes that leak through
    name = re.sub(r"^\d+[\s/½¼¾⅓⅔⅛]*\s*", "", name)
    # Normalize whitespace
    name = re.sub(r"\s+", " ", name)
    # Remove trailing "of" artifacts
    name = re.sub(r"\s+of$", "", name)
    return name


def extract_micronutrients_from_text(text: str) -> list[str]:
    """
    Extract micronutrient entity names from MicronutrientHealthFunctions text.

    The text contains multiple descriptions separated by newlines,
    each wrapped in quotes within a list. We match against our
    known description→nutrient mapping.
    """
    if not text or str(text).strip() in ("", "nan", "None", "[]"):
        return []

    text_lower = str(text).lower()
    found = set()

    for description_key, nutrient_name in MICRO_DESCRIPTION_TO_NUTRIENT.items():
        if description_key.lower() in text_lower:
            found.add(nutrient_name)

    return sorted(found)


def extract_health_functions_from_text(text: str) -> list[str]:
    """
    Extract short health function labels from HealthFunctions text.
    """
    if not text or str(text).strip() in ("", "nan", "None", "[]"):
        return []

    text_lower = str(text).lower()
    found = set()

    for description_key, label in HEALTH_DESCRIPTION_TO_LABEL.items():
        if description_key.lower() in text_lower:
            found.add(label)

    return sorted(found)


def safe_float(val) -> float | None:
    """Safely parse a float, return None on failure."""
    if val is None:
        return None
    try:
        f = float(str(val).strip())
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════
#  SECTION 7: MAIN TRIPLE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_triples_from_csv(input_path: str, output_path: str):
    """
    Read df_foodcom_recipes_filtered.csv and produce triples.tsv.

    For ~500K recipes, this will generate millions of triples.
    We use pandas for parsing because the CSV has multiline fields
    (HealthFunctions spans multiple lines inside quotes).
    """
    import pandas as pd

    print("=" * 70)
    print("VitalBites CSV → Knowledge Graph Triples")
    print("=" * 70)

    # ── Load CSV ──────────────────────────────────────────────────
    # pandas handles the multiline quoted fields correctly
    print(f"\nLoading {input_path}...")
    print("  (This may take a minute for 9.7M lines — the multiline fields")
    print("   mean ~500K actual recipes, pandas will handle it)")

    # For very large files, we can use chunked reading
    # But first let's try loading it all — 500K recipes should fit in RAM
    try:
        df = pd.read_csv(
            input_path,
            dtype=str,  # Read everything as strings first for safe parsing
            on_bad_lines="skip",  # Skip malformed rows
            engine="python",  # Python engine handles multiline fields better
        )
    except Exception as e:
        print(f"  Full load failed ({e}), trying chunked approach...")
        df = chunked_load(input_path)

    print(f"  Loaded {len(df):,} recipes")
    print(f"  Columns: {list(df.columns)}")

    # ── Extract triples ───────────────────────────────────────────
    triples = set()  # Use set for automatic dedup
    recipe_count = 0
    skipped = 0

    # Stats tracking
    field_coverage = defaultdict(int)

    for idx, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name or name == "nan":
            skipped += 1
            continue

        # ── Ingredients ───────────────────────────────────────────
        ingredients = parse_numpy_array(row.get("cleaned_ingredients", ""))
        if not ingredients:
            # Try RecipeIngredientParts as fallback
            ingredients = parse_numpy_array(row.get("RecipeIngredientParts", ""))
        if not ingredients:
            skipped += 1
            continue

        recipe_count += 1
        field_coverage["ingredients"] += 1

        for ing in ingredients:
            if ing:  # skip empty strings
                triples.add((name, "CONTAINS_INGREDIENT", ing))

        # ── Category ──────────────────────────────────────────────
        category = str(row.get("RecipeCategory", "")).strip()
        if category and category != "nan":
            triples.add((name, "IN_CATEGORY", category))
            field_coverage["category"] += 1

        # ── Macronutrient thresholds (per serving) ────────────────
        for col, (nutrient_label, threshold) in MACRO_THRESHOLDS.items():
            val = safe_float(row.get(col))
            if val is not None and val >= threshold:
                triples.add((name, "HIGH_IN", nutrient_label))
                field_coverage[f"high_in_{nutrient_label}"] += 1

        # Calorie classification
        cal_per_serving = safe_float(row.get("Calories_per_serving"))
        if cal_per_serving is not None:
            for label, (low, high) in CALORIE_THRESHOLDS.items():
                if low <= cal_per_serving < high:
                    triples.add((name, "IS", label))
                    field_coverage[label] += 1
                    break

        # ── Health Functions (text → short labels) ────────────────
        hf_text = str(row.get("HealthFunctions", ""))
        health_labels = extract_health_functions_from_text(hf_text)
        if health_labels:
            field_coverage["health_functions"] += 1
        for label in health_labels:
            triples.add((name, "HAS_HEALTH_FUNCTION", label))

        # ── Micronutrients (text → nutrient entities) ─────────────
        micro_text = str(row.get("MicronutrientHealthFunctions", ""))
        micronutrients = extract_micronutrients_from_text(micro_text)
        if micronutrients:
            field_coverage["micronutrients"] += 1
        for nutrient in micronutrients:
            triples.add((name, "PROVIDES", nutrient))

        # ── Progress ──────────────────────────────────────────────
        if recipe_count % 50000 == 0:
            print(f"  Processed {recipe_count:,} recipes, {len(triples):,} triples so far...")

    # ── Substitution triples (curated, not from CSV) ──────────────
    print(f"\nAdding substitution triples...")
    sub_count = 0
    for ingredient, subs in SUBSTITUTIONS.items():
        for sub in subs:
            triples.add((ingredient, "SUBSTITUTES_FOR", sub))
            triples.add((sub, "SUBSTITUTES_FOR", ingredient))
            sub_count += 2

    # ── Ailment → Micronutrient triples (curated) ─────────────────
    print(f"Adding ailment → micronutrient triples...")
    ailment_count = 0
    for ailment, info in AILMENT_NUTRIENT_MAP.items():
        for nutrient in info["needs"]:
            triples.add((ailment, "BENEFITS_FROM", nutrient))
            ailment_count += 1
        for nutrient in info["avoid"]:
            triples.add((ailment, "SHOULD_AVOID", nutrient))
            ailment_count += 1

    # ── Write output ──────────────────────────────────────────────
    triples_list = sorted(triples, key=lambda t: (t[1], t[0], t[2]))

    print(f"\nWriting {len(triples_list):,} triples to {output_path}...")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for h, r, t in triples_list:
            writer.writerow([h, r, t])

    # ── Stats ─────────────────────────────────────────────────────
    relation_counts = Counter(r for _, r, _ in triples_list)
    entities = set()
    for h, _, t in triples_list:
        entities.add(h)
        entities.add(t)

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Recipes processed:   {recipe_count:,}")
    print(f"  Recipes skipped:     {skipped:,} (no name or no ingredients)")
    print(f"  Total triples:       {len(triples_list):,}")
    print(f"  Unique entities:     {len(entities):,}")
    print(f"  Unique relations:    {len(relation_counts)}")

    print(f"\n  TRIPLES PER RELATION TYPE:")
    for rel, count in relation_counts.most_common():
        print(f"    {rel:30s} {count:>10,}")

    print(f"\n  FIELD COVERAGE (of {recipe_count:,} recipes):")
    print(f"    ingredients:       {field_coverage['ingredients']:>10,} ({100*field_coverage['ingredients']/max(recipe_count,1):.1f}%)")
    print(f"    category:          {field_coverage['category']:>10,} ({100*field_coverage['category']/max(recipe_count,1):.1f}%)")
    print(f"    health_functions:  {field_coverage['health_functions']:>10,} ({100*field_coverage['health_functions']/max(recipe_count,1):.1f}%)")
    print(f"    micronutrients:    {field_coverage['micronutrients']:>10,} ({100*field_coverage['micronutrients']/max(recipe_count,1):.1f}%)")

    print(f"\n  Substitution triples: {sub_count:,}")
    print(f"  Ailment triples:      {ailment_count:,}")
    print(f"\n  Output: {output_path}")
    print(f"\n  Next step:")
    print(f"    python train_and_infer.py train --triples {output_path} --epochs 300 --dim 256")


def chunked_load(input_path: str, chunksize: int = 50000):
    """
    Fallback: Load CSV in chunks for very large files.
    Concatenates all chunks into a single DataFrame.
    """
    import pandas as pd

    chunks = []
    total = 0
    for chunk in pd.read_csv(
        input_path,
        dtype=str,
        on_bad_lines="skip",
        engine="python",
        chunksize=chunksize,
    ):
        chunks.append(chunk)
        total += len(chunk)
        print(f"  Loaded chunk: {total:,} rows so far...")

    return pd.concat(chunks, ignore_index=True)


# ══════════════════════════════════════════════════════════════════
#  SECTION 8: CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VitalBites: CSV → Knowledge Graph Triples",
        epilog="""
Full pipeline:
    python csv_to_triples_vitalbites.py --input df_foodcom_recipes_filtered.csv
    python train_and_infer.py train --triples triples.tsv --epochs 300 --dim 256
    python train_and_infer.py recommend --ailment "anemia" --top 10
        """,
    )
    parser.add_argument(
        "--input",
        default="df_foodcom_recipes_filtered.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default="triples.tsv",
        help="Output triples TSV path",
    )
    args = parser.parse_args()

    extract_triples_from_csv(args.input, args.output)
