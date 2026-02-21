"""
VitalBites CSV → Knowledge Graph Triples Pipeline (v2)
======================================================
Converts df_foodcom_recipes_filtered.csv into triples.tsv for RotatE training.

v2 Changes:
  - Loads all mappings from mined_config.json (produced by mine_knowledge.py)
  - Normalizes ingredient names to canonical forms
  - Normalizes nutrient names to canonical forms
  - Uses mined ailment mappings (50+ conditions vs 14 hardcoded)
  - Uses validated substitutions (hundreds vs 34 hardcoded)
  - Falls back to built-in defaults if mined_config.json not found

Pipeline:
    # Step 1: Mine knowledge (run once)
    python mine_knowledge.py all --input df_foodcom_recipes_filtered.csv --api-key sk-ant-...

    # Step 2: Extract triples (uses mined_config.json automatically)
    python csv_to_triples_vitalbites.py --input df_foodcom_recipes_filtered.csv --output triples.tsv

    # Step 3: Train on Colab
    python train_and_infer.py train --triples triples.tsv --epochs 300 --dim 256

Requirements:
    pip install pandas

Author: Brian / LeeroyChainkins AI — VitalBites Project
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION LOADER
# ══════════════════════════════════════════════════════════════════

# Built-in fallback defaults (used only if mined_config.json is missing)
_FALLBACK_MICRO = {
    "vital for oxygen delivery in blood, energy metabolism, and immune system function": "iron",
    "vital for redox reactions, energy production, and cellular respiration": "copper",
    "helps with iron absorption, energy production, and the formation of collagen and neurotransmitters": "copper",
    "functions as an antioxidant, supports collagen synthesis, aids immune cell function, enhances iron absorption, and supports neurotransmitter synthesis": "vitamin_c",
    "essential for dna synthesis, cell division, fetal development, and the prevention of neural tube defects": "folate",
    "supports bone and teeth strength, muscle function, nerve signaling, and blood coagulation": "calcium",
    "essential for bone mineralization, energy storage and transfer, and cell membrane integrity": "phosphorus",
    "supports immune health, cellular growth, enzyme function, and wound repair": "zinc",
    "plays a role in enzymatic reactions including energy metabolism, muscle and nerve function, and dna synthesis": "magnesium",
    "maintains normal heart rhythm, muscle contractions, and nerve impulses by balancing fluids and electrolytes": "potassium",
    "controls blood volume and pressure, supports nerve impulse transmission and muscle contraction": "sodium",
    "protects cells from oxidative damage and supports thyroid hormone metabolism": "selenium",
    "protects membranes from oxidative stress, modulates inflammation, supports the immune system, and may have anti-cancer effects": "vitamin_e",
    "crucial for vision, immune response, skin health, and cellular differentiation": "vitamin_a",
    "regulates calcium and phosphorus homeostasis, supports bone health, and contributes to immune function": "vitamin_d",
    "activates clotting factors and is essential for blood coagulation and bone metabolism": "vitamin_k1",
    "cofactor for gamma-carboxylation of proteins, supports bone mineralization, vascular health, myelin repair, and has anti-inflammatory effects": "vitamin_k2",
    "vital for red blood cell production, neurological function, and dna synthesis": "vitamin_b12",
    "they are vital for red blood cell production, neurological function, and dna synthesis": "vitamin_b12",
    "coenzyme in over 140 reactions, including amino acid metabolism, neurotransmitter synthesis, hemoglobin production, and supports the immune system": "vitamin_b6",
    "cofactor for enzymes involved in carbohydrate metabolism, supports energy production, nerve function, and mitochondrial health": "thiamin",
    "central role in redox reactions, energy production, dna repair, nervous system function, and maintaining the skin barrier": "niacin",
    "required for fatty acid metabolism, synthesis of coenzyme a, and energy production": "pantothenic_acid",
    "important for fatty acid synthesis, gluconeogenesis, and amino acid metabolism": "biotin",
    "involved in bone formation, metabolism of carbohydrates and amino acids, and antioxidant protection": "manganese",
    "regulates metabolism, growth, and development via thyroid hormone synthesis": "iodine",
    "fatty acids serve as a key energy source, components of triglycerides and phospholipids": "fatty_acids",
    "acts as a coenzyme in oxidation-reduction reactions, supports energy production": "riboflavin",
}

_FALLBACK_HEALTH = {
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

_FALLBACK_AILMENTS = {
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

_FALLBACK_SUBSTITUTIONS = {}  # empty — the mined ones are much better


def load_config(config_path: str) -> dict:
    """
    Load mined_config.json. Falls back to built-in defaults if not found.
    """
    config = {
        "MICRO_DESCRIPTION_TO_NUTRIENT": _FALLBACK_MICRO,
        "HEALTH_DESCRIPTION_TO_LABEL": _FALLBACK_HEALTH,
        "AILMENT_NUTRIENT_MAP": _FALLBACK_AILMENTS,
        "SUBSTITUTIONS": _FALLBACK_SUBSTITUTIONS,
        "INGREDIENT_NORMALIZE": {},
        "NUTRIENT_NORMALIZE": {},
    }

    if os.path.exists(config_path):
        print(f"  Loading mined config from {config_path}...")
        with open(config_path) as f:
            mined = json.load(f)

        # Override defaults with mined data (only if present and non-empty)
        for key in config:
            if key in mined and mined[key]:
                config[key] = mined[key]
                print(f"    {key}: {len(mined[key])} entries (mined)")
            else:
                print(f"    {key}: {len(config[key])} entries (fallback)")
    else:
        print(f"  WARNING: {config_path} not found — using built-in fallback defaults.")
        print(f"           Run mine_knowledge.py first for better results.")

    return config


# ══════════════════════════════════════════════════════════════════
#  MACRONUTRIENT THRESHOLDS (not mined — these are domain constants)
# ══════════════════════════════════════════════════════════════════

MACRO_THRESHOLDS = {
    "ProteinContent_per_serving": ("protein", 15.0),
    "FiberContent_per_serving": ("fiber", 5.0),
    "FatContent_per_serving": ("fat", 20.0),
    "SaturatedFatContent_per_serving": ("saturated_fat", 10.0),
    "SugarContent_per_serving": ("sugar", 15.0),
    "SodiumContent_per_serving": ("sodium_high", 400.0),
}

CALORIE_THRESHOLDS = {
    "low-calorie": (0, 200),
    "moderate-calorie": (200, 400),
    "high-calorie": (400, float("inf")),
}


# ══════════════════════════════════════════════════════════════════
#  PARSING UTILITIES
# ══════════════════════════════════════════════════════════════════

def parse_numpy_array(raw: str, ingredient_normalize: dict = None) -> list[str]:
    """
    Parse numpy-style array strings and optionally normalize ingredient names.
    """
    if not raw or str(raw).strip() in ("", "nan", "None", "[]", "['']"):
        return []

    raw = str(raw).strip()
    items = re.findall(r"'([^']*?)'", raw)
    if not items:
        items = re.findall(r'"([^"]*?)"', raw)

    result = []
    for i in items:
        name = basic_normalize(i)
        if not name:
            continue
        # Apply mined normalization if available
        if ingredient_normalize and name in ingredient_normalize:
            name = ingredient_normalize[name]
        result.append(name)

    return result


def basic_normalize(name: str) -> str:
    """Basic ingredient normalization (always applied)."""
    name = name.lower().strip()
    name = re.sub(r"^\d+[\s/½¼¾⅓⅔⅛]*\s*", "", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\s+of$", "", name)
    return name


def extract_micronutrients(text: str, micro_map: dict, nutrient_normalize: dict) -> list[str]:
    """Extract and normalize micronutrient names from description text."""
    if not text or str(text).strip() in ("", "nan", "None", "[]"):
        return []

    text_lower = str(text).lower()
    found = set()

    for description_key, nutrient_name in micro_map.items():
        if description_key.lower() in text_lower:
            # Apply nutrient normalization
            canonical = nutrient_normalize.get(nutrient_name, nutrient_name)
            if canonical:  # skip nulls
                found.add(canonical)

    return sorted(found)


def extract_health_functions(text: str, health_map: dict) -> list[str]:
    """Extract short health function labels from description text."""
    if not text or str(text).strip() in ("", "nan", "None", "[]"):
        return []

    text_lower = str(text).lower()
    found = set()

    for description_key, label in health_map.items():
        if description_key.lower() in text_lower:
            found.add(label)

    return sorted(found)


def safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(str(val).strip())
        return f if f == f else None
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════
#  MAIN TRIPLE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_triples_from_csv(input_path: str, output_path: str, config_path: str):
    """
    Read CSV and produce triples.tsv using mined_config.json for all mappings.
    """
    import pandas as pd

    print("=" * 70)
    print("VitalBites CSV → Knowledge Graph Triples (v2)")
    print("=" * 70)

    # ── Load config ───────────────────────────────────────────────
    config = load_config(config_path)
    micro_map = config["MICRO_DESCRIPTION_TO_NUTRIENT"]
    health_map = config["HEALTH_DESCRIPTION_TO_LABEL"]
    ailment_map = config["AILMENT_NUTRIENT_MAP"]
    substitutions = config["SUBSTITUTIONS"]
    ingredient_normalize = config["INGREDIENT_NORMALIZE"]
    nutrient_normalize = config["NUTRIENT_NORMALIZE"]

    # ── Load CSV ──────────────────────────────────────────────────
    print(f"\nLoading {input_path}...")
    try:
        df = pd.read_csv(
            input_path, dtype=str, on_bad_lines="skip", engine="python",
        )
    except Exception as e:
        print(f"  Full load failed ({e}), trying chunked approach...")
        df = chunked_load(input_path)

    print(f"  Loaded {len(df):,} recipes")

    # ── Extract triples ───────────────────────────────────────────
    triples = set()
    recipe_count = 0
    skipped = 0
    field_coverage = defaultdict(int)

    for idx, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name or name == "nan":
            skipped += 1
            continue

        # ── Ingredients (with normalization) ──────────────────────
        ingredients = parse_numpy_array(
            row.get("cleaned_ingredients", ""),
            ingredient_normalize=ingredient_normalize,
        )
        if not ingredients:
            ingredients = parse_numpy_array(
                row.get("RecipeIngredientParts", ""),
                ingredient_normalize=ingredient_normalize,
            )
        if not ingredients:
            skipped += 1
            continue

        recipe_count += 1
        field_coverage["ingredients"] += 1

        for ing in ingredients:
            if ing:
                triples.add((name, "CONTAINS_INGREDIENT", ing))

        # ── Category ──────────────────────────────────────────────
        category = str(row.get("RecipeCategory", "")).strip()
        if category and category != "nan":
            triples.add((name, "IN_CATEGORY", category))
            field_coverage["category"] += 1

        # ── Macronutrient thresholds ──────────────────────────────
        for col, (nutrient_label, threshold) in MACRO_THRESHOLDS.items():
            val = safe_float(row.get(col))
            if val is not None and val >= threshold:
                triples.add((name, "HIGH_IN", nutrient_label))
                field_coverage[f"high_in_{nutrient_label}"] += 1

        # Calorie classification
        cal = safe_float(row.get("Calories_per_serving"))
        if cal is not None:
            for label, (low, high) in CALORIE_THRESHOLDS.items():
                if low <= cal < high:
                    triples.add((name, "IS", label))
                    field_coverage[label] += 1
                    break

        # ── Health Functions (text → short labels, from mined config) ─
        hf_text = str(row.get("HealthFunctions", ""))
        health_labels = extract_health_functions(hf_text, health_map)
        if health_labels:
            field_coverage["health_functions"] += 1
        for label in health_labels:
            triples.add((name, "HAS_HEALTH_FUNCTION", label))

        # ── Micronutrients (text → normalized nutrient entities) ──
        micro_text = str(row.get("MicronutrientHealthFunctions", ""))
        micronutrients = extract_micronutrients(micro_text, micro_map, nutrient_normalize)
        if micronutrients:
            field_coverage["micronutrients"] += 1
        for nutrient in micronutrients:
            triples.add((name, "PROVIDES", nutrient))

        # ── Progress ──────────────────────────────────────────────
        if recipe_count % 50000 == 0:
            print(f"  Processed {recipe_count:,} recipes, {len(triples):,} triples so far...")

    # ── Substitution triples (from mined config) ──────────────────
    print(f"\nAdding substitution triples...")
    sub_count = 0
    for ingredient, subs in substitutions.items():
        for sub in subs:
            # Normalize both sides
            ing_norm = ingredient_normalize.get(ingredient, ingredient)
            sub_norm = ingredient_normalize.get(sub, sub)
            if ing_norm != sub_norm:  # don't self-substitute
                triples.add((ing_norm, "SUBSTITUTES_FOR", sub_norm))
                triples.add((sub_norm, "SUBSTITUTES_FOR", ing_norm))
                sub_count += 2

    # ── Ailment → Micronutrient triples (from mined config) ───────
    print(f"Adding ailment → micronutrient triples...")
    ailment_count = 0
    for ailment, info in ailment_map.items():
        for nutrient in info.get("needs", []):
            # Normalize nutrient name
            canonical = nutrient_normalize.get(nutrient, nutrient)
            if canonical:
                triples.add((ailment, "BENEFITS_FROM", canonical))
                ailment_count += 1
        for nutrient in info.get("avoid", []):
            canonical = nutrient_normalize.get(nutrient, nutrient)
            if canonical:
                triples.add((ailment, "SHOULD_AVOID", canonical))
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

    using_mined = os.path.exists(config_path)
    if using_mined:
        print(f"\n  Config source: {config_path} (mined)")
    else:
        print(f"\n  Config source: built-in fallbacks (run mine_knowledge.py for better results)")

    print(f"\n  Output: {output_path}")
    print(f"\n  Next step:")
    print(f"    python train_and_infer.py train --triples {output_path} --epochs 300 --dim 256")


def chunked_load(input_path: str, chunksize: int = 50000):
    """Fallback: Load CSV in chunks for very large files."""
    import pandas as pd

    chunks = []
    total = 0
    for chunk in pd.read_csv(
        input_path, dtype=str, on_bad_lines="skip", engine="python",
        chunksize=chunksize,
    ):
        chunks.append(chunk)
        total += len(chunk)
        print(f"  Loaded chunk: {total:,} rows so far...")

    return pd.concat(chunks, ignore_index=True)


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VitalBites: CSV → Knowledge Graph Triples (v2)",
        epilog="""
Full pipeline:
    python mine_knowledge.py all --input df_foodcom_recipes_filtered.csv --api-key sk-ant-...
    python csv_to_triples_vitalbites.py --input df_foodcom_recipes_filtered.csv
    python train_and_infer.py train --triples triples.tsv --epochs 300 --dim 256
    python train_and_infer.py recommend --ailment "anemia" --top 10
        """,
    )
    parser.add_argument("--input", default="df_foodcom_recipes_filtered.csv", help="Input CSV")
    parser.add_argument("--output", default="triples.tsv", help="Output triples TSV")
    parser.add_argument("--config", default="mined_config.json", help="Mined config JSON path")
    args = parser.parse_args()

    extract_triples_from_csv(args.input, args.output, args.config)

