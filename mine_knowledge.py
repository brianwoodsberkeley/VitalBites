"""
VitalBites Knowledge Miner
===========================
Mines your actual CSV data to automatically discover:
  1. AILMENTS — extracted from HealthFunctions & MicronutrientHealthFunctions text
  2. SUBSTITUTIONS — discovered from ingredient co-occurrence patterns
  3. NUTRIENT→AILMENT mappings — which nutrients help which conditions
  4. HEALTH FUNCTION LABELS — short labels from the long description texts
  5. NORMALIZATION — canonical names for ingredients, nutrients, and substitutions

This replaces the hand-curated dictionaries in csv_to_triples_vitalbites.py
with data-driven ones mined from your 312K recipes.

Commands:
  python mine_knowledge.py extract        → Pull unique texts from CSV (no API)
  python mine_knowledge.py enrich         → Claude maps descriptions → structured data
  python mine_knowledge.py normalize      → Claude normalizes ingredients + nutrients
  python mine_knowledge.py substitutions  → Co-occurrence based substitute discovery
  python mine_knowledge.py config         → Combine all outputs into mined_config.json
  python mine_knowledge.py all            → Everything in one shot

Output:
  mined_ailments.json              — ailment → [nutrients] mapping
  mined_substitutions.json         — ingredient → [substitutes] mapping
  mined_substitutions_simple.json  — simple ingredient → [substitutes] 
  mined_health_labels.json         — long description → short label mapping
  mined_nutrient_map.json          — description → nutrient name mapping
  mined_ingredient_normalize.json  — raw ingredient → canonical name mapping
  mined_nutrient_normalize.json    — raw nutrient name → canonical name mapping
  mined_config.json                — combined config for csv_to_triples_vitalbites.py

Then use mined_config.json in csv_to_triples_vitalbites.py and re-extract triples.

Requirements:
    pip install pandas numpy anthropic

Author: Brian / LeeroyChainkins AI — VitalBites Project
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_json_response(text: str) -> dict | list:
    """
    Robustly parse JSON from Claude's response.
    Handles markdown code fences, preamble text, and other wrapping.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Find the first { or [ and last } or ]
    first_brace = None
    last_brace = None
    for i, c in enumerate(text):
        if c in "{[" and first_brace is None:
            first_brace = i
        if c in "}]":
            last_brace = i

    if first_brace is not None and last_brace is not None:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Last resort: dump raw text for debugging
    print(f"    WARNING: Could not parse JSON from response.")
    print(f"    First 500 chars: {text[:500]}")
    raise ValueError("Failed to parse JSON from Claude response")


# ══════════════════════════════════════════════════════════════════
#  PART 1: EXTRACT UNIQUE TEXTS FROM CSV
# ══════════════════════════════════════════════════════════════════

def extract_unique_texts(input_path: str):
    """
    Scan the CSV and collect all unique text descriptions from
    HealthFunctions and MicronutrientHealthFunctions columns.

    These are the raw material Claude will process into structured mappings.
    """
    import pandas as pd

    print("=" * 70)
    print("STEP 1: Extracting unique texts from CSV")
    print("=" * 70)

    print(f"\nLoading {input_path}...")
    df = pd.read_csv(input_path, dtype=str, on_bad_lines="skip", engine="python")
    print(f"  Loaded {len(df):,} rows")

    # ── Extract unique health function descriptions ───────────────
    print("\nExtracting unique HealthFunction descriptions...")
    health_texts = set()
    for raw in df["HealthFunctions"].dropna():
        # Each cell contains multiple descriptions in a list
        descriptions = re.findall(r"'([^']{20,}?)'", str(raw))
        for desc in descriptions:
            health_texts.add(desc.strip().lower())

    print(f"  Found {len(health_texts)} unique health function descriptions")

    # ── Extract unique micronutrient descriptions ─────────────────
    print("Extracting unique MicronutrientHealthFunction descriptions...")
    micro_texts = set()
    for raw in df["MicronutrientHealthFunctions"].dropna():
        descriptions = re.findall(r"'([^']{20,}?)'", str(raw))
        for desc in descriptions:
            micro_texts.add(desc.strip().lower())

    print(f"  Found {len(micro_texts)} unique micronutrient descriptions")

    # ── Extract unique categories ─────────────────────────────────
    print("Extracting unique categories...")
    categories = set()
    for raw in df["RecipeCategory"].dropna():
        cat = str(raw).strip()
        if cat and cat != "nan":
            categories.add(cat)

    print(f"  Found {len(categories)} unique categories")

    # ── Extract unique ingredients ────────────────────────────────
    print("Extracting unique ingredients...")
    ingredient_counter = Counter()
    recipe_ingredients = []  # list of sets, for co-occurrence mining

    for raw in df["cleaned_ingredients"].dropna():
        items = re.findall(r"'([^']*?)'", str(raw))
        cleaned = set()
        for item in items:
            ing = item.strip().lower()
            ing = re.sub(r"\s+", " ", ing)
            if ing and len(ing) > 1:
                ingredient_counter[ing] += 1
                cleaned.add(ing)
        if cleaned:
            recipe_ingredients.append(cleaned)

    print(f"  Found {len(ingredient_counter)} unique ingredients")
    print(f"  Top 20 ingredients:")
    for ing, count in ingredient_counter.most_common(20):
        print(f"    {ing:30s} {count:>8,}")

    # ── Save extracted data ───────────────────────────────────────
    extracted = {
        "health_function_texts": sorted(health_texts),
        "micronutrient_texts": sorted(micro_texts),
        "categories": sorted(categories),
        "ingredient_counts": dict(ingredient_counter.most_common(5000)),  # top 5K
        "total_recipes": len(df),
        "total_unique_ingredients": len(ingredient_counter),
    }

    with open("extracted_texts.json", "w") as f:
        json.dump(extracted, f, indent=2)

    print(f"\n  Saved to extracted_texts.json")
    print(f"  Health function texts:  {len(health_texts)}")
    print(f"  Micronutrient texts:    {len(micro_texts)}")
    print(f"  Categories:             {len(categories)}")
    print(f"  Ingredients (top 5K):   {min(5000, len(ingredient_counter))}")

    return extracted, recipe_ingredients


# ══════════════════════════════════════════════════════════════════
#  PART 2: USE CLAUDE TO GENERATE MAPPINGS
# ══════════════════════════════════════════════════════════════════

def enrich_with_claude(api_key: str):
    """
    Send the extracted unique texts to Claude and get back:
      1. Micronutrient description → nutrient name mapping
      2. Nutrient → ailment mappings
      3. Health function description → short label mapping
    """
    import anthropic

    print("\n" + "=" * 70)
    print("STEP 2: Enriching with Claude API")
    print("=" * 70)

    # Load extracted texts
    with open("extracted_texts.json") as f:
        extracted = json.load(f)

    client = anthropic.Anthropic(api_key=api_key)

    # ── CALL 1: Map micronutrient descriptions → nutrient names ───
    print("\n  Call 1: Mapping micronutrient descriptions to nutrient names...")

    micro_texts = extracted["micronutrient_texts"]
    prompt_1 = f"""I have a recipe nutrition database. The following are ALL unique text descriptions 
from a "MicronutrientHealthFunctions" column. Each description describes what a specific 
micronutrient does in the body.

Your job: For EACH description, identify which micronutrient it's describing.

Return ONLY a JSON object where:
- Keys are the exact description text (lowercase, as provided)
- Values are the standardized nutrient name using this format:
  iron, zinc, calcium, magnesium, potassium, sodium, phosphorus, selenium, manganese,
  iodine, copper, chromium, molybdenum, fluoride,
  vitamin_a, vitamin_b1, vitamin_b2, vitamin_b3, vitamin_b5, vitamin_b6, vitamin_b7,
  vitamin_b9, vitamin_b12, vitamin_c, vitamin_d, vitamin_e, vitamin_k1, vitamin_k2,
  omega3, omega6, fatty_acids, choline, protein, fiber, carbohydrates, fat

If a description matches multiple nutrients, pick the PRIMARY one it's describing.
If you're unsure, use your best judgment based on the biological function described.

Here are the descriptions:
{json.dumps(micro_texts, indent=2)}

Return ONLY valid JSON, no other text."""

    response_1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt_1}],
    )
    nutrient_map = parse_json_response(response_1.content[0].text)

    with open("mined_nutrient_map.json", "w") as f:
        json.dump(nutrient_map, f, indent=2)
    print(f"    Mapped {len(nutrient_map)} descriptions → nutrient names")
    print(f"    Unique nutrients found: {len(set(nutrient_map.values()))}")
    print(f"    Saved to mined_nutrient_map.json")

    # ── CALL 2: Map health function descriptions → short labels ───
    print("\n  Call 2: Mapping health function descriptions to short labels...")

    health_texts = extracted["health_function_texts"]
    prompt_2 = f"""I have a recipe nutrition database. The following are ALL unique text descriptions
from a "HealthFunctions" column. Each describes a macro-level health benefit.

Your job: For EACH description, create a SHORT label (2-4 words, hyphenated) that
captures the core health function. These will be used as entity names in a knowledge graph.

Return ONLY a JSON object where:
- Keys are the exact description text (lowercase, as provided)
- Values are short hyphenated labels like: "heart-health", "bone-strength", 
  "immune-support", "energy-production", "digestive-health", "blood-sugar-control",
  "anti-inflammatory", "brain-health", "muscle-function", "skin-health", etc.

Multiple descriptions CAN map to the same label if they describe the same function.

Here are the descriptions:
{json.dumps(health_texts, indent=2)}

Return ONLY valid JSON, no other text."""

    response_2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt_2}],
    )
    health_label_map = parse_json_response(response_2.content[0].text)

    with open("mined_health_labels.json", "w") as f:
        json.dump(health_label_map, f, indent=2)
    print(f"    Mapped {len(health_label_map)} descriptions → short labels")
    print(f"    Unique labels: {len(set(health_label_map.values()))}")
    print(f"    Saved to mined_health_labels.json")

    # ── CALL 3: Generate comprehensive ailment → nutrient mapping ─
    print("\n  Call 3: Generating ailment → nutrient mappings...")

    unique_nutrients = sorted(set(nutrient_map.values()))
    prompt_3 = f"""You are a clinical nutritionist. I'm building a recipe recommendation system 
that suggests recipes based on a user's health conditions.

Here are the micronutrients available in my database:
{json.dumps(unique_nutrients, indent=2)}

Generate a comprehensive mapping of health conditions/ailments to the micronutrients 
that HELP with that condition, and any that should be AVOIDED or LIMITED.

Include at least 50 conditions covering:
- Common deficiencies (anemia, osteoporosis, etc.)
- Chronic diseases (type 2 diabetes, hypertension, heart disease, etc.)  
- Mental health (depression, anxiety, insomnia, brain fog, etc.)
- Immune conditions (weak immunity, frequent colds, autoimmune support, etc.)
- Digestive issues (IBS, constipation, acid reflux, etc.)
- Women's health (PMS, pregnancy support, menopause, PCOS, etc.)
- Men's health (prostate health, low testosterone support, etc.)
- Aging (cognitive decline, sarcopenia, macular degeneration, etc.)
- Skin/hair/nails (acne, eczema, hair loss, brittle nails, etc.)
- Energy/performance (chronic fatigue, athletic recovery, etc.)
- Bone/joint (osteoarthritis, fracture recovery, etc.)
- Cardiovascular (high cholesterol, poor circulation, etc.)

Return ONLY a JSON object where:
- Keys are condition names (lowercase, underscore-separated)
- Values are objects with "needs" (list of helpful nutrients) and "avoid" (list to limit)
- Only use nutrient names from the list I provided above

Be thorough — more conditions = better recommendations for our users.

Return ONLY valid JSON, no other text."""

    response_3 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt_3}],
    )
    ailment_map = parse_json_response(response_3.content[0].text)

    with open("mined_ailments.json", "w") as f:
        json.dump(ailment_map, f, indent=2)
    print(f"    Generated {len(ailment_map)} ailment mappings")
    print(f"    Saved to mined_ailments.json")

    # ── Summary ───────────────────────────────────────────────────
    total_needs = sum(len(v.get("needs", [])) for v in ailment_map.values())
    total_avoids = sum(len(v.get("avoid", [])) for v in ailment_map.values())

    print(f"\n{'='*70}")
    print(f"ENRICHMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Nutrient mappings:   {len(nutrient_map)} descriptions → {len(set(nutrient_map.values()))} nutrients")
    print(f"  Health labels:       {len(health_label_map)} descriptions → {len(set(health_label_map.values()))} labels")
    print(f"  Ailment mappings:    {len(ailment_map)} conditions")
    print(f"    Total BENEFITS_FROM edges: {total_needs}")
    print(f"    Total SHOULD_AVOID edges:  {total_avoids}")
    print(f"\n  Output files:")
    print(f"    mined_nutrient_map.json")
    print(f"    mined_health_labels.json")
    print(f"    mined_ailments.json")

    return nutrient_map, health_label_map, ailment_map


# ══════════════════════════════════════════════════════════════════
#  PART 3: MINE SUBSTITUTIONS FROM CO-OCCURRENCE
# ══════════════════════════════════════════════════════════════════

def mine_substitutions(input_path: str, min_recipes: int = 50, top_k: int = 5):
    """
    Discover ingredient substitutions from co-occurrence patterns.

    Logic: Ingredients that appear in similar RECIPE CONTEXTS but rarely
    appear TOGETHER in the same recipe are likely substitutes.

    Example: "butter" and "olive oil" both appear in thousands of recipes
    with flour, sugar, eggs, etc. — but they rarely appear in the SAME recipe.
    That's a substitution signal.

    We use Pointwise Mutual Information (PMI) with a twist:
      - High PMI with shared neighbors = similar role
      - Low co-occurrence in same recipe = substitutability
    """
    import pandas as pd
    import numpy as np

    print("\n" + "=" * 70)
    print("STEP 3: Mining ingredient substitutions from co-occurrence")
    print("=" * 70)

    print(f"\nLoading {input_path}...")
    df = pd.read_csv(input_path, dtype=str, on_bad_lines="skip", engine="python")

    # Parse all ingredient sets
    print("Parsing ingredients...")
    ingredient_counter = Counter()
    recipe_sets = []
    for raw in df["cleaned_ingredients"].dropna():
        items = re.findall(r"'([^']*?)'", str(raw))
        cleaned = set()
        for item in items:
            ing = item.strip().lower()
            ing = re.sub(r"\s+", " ", ing)
            if ing and len(ing) > 1:
                ingredient_counter[ing] += 1
                cleaned.add(ing)
        if cleaned:
            recipe_sets.append(frozenset(cleaned))

    total_recipes = len(recipe_sets)
    print(f"  {total_recipes:,} recipes with ingredients")

    # Focus on ingredients that appear in at least min_recipes
    common_ingredients = {
        ing for ing, count in ingredient_counter.items() if count >= min_recipes
    }
    print(f"  {len(common_ingredients)} ingredients with >= {min_recipes} recipes")

    # Build co-occurrence matrix (sparse, using dicts)
    print("Building co-occurrence matrix...")
    cooccur = defaultdict(Counter)      # ing_a → {ing_b: count}
    ingredient_recipes = defaultdict(int)  # ing → recipe count

    for recipe in recipe_sets:
        common_in_recipe = recipe & common_ingredients
        for ing in common_in_recipe:
            ingredient_recipes[ing] += 1
        common_list = sorted(common_in_recipe)
        for i, a in enumerate(common_list):
            for b in common_list[i + 1:]:
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    # Find substitutes: ingredients with similar "neighbor profiles"
    # but low direct co-occurrence
    print("Computing substitution scores...")

    # For each ingredient, get its "context" = set of ingredients it commonly appears with
    context = {}
    for ing in common_ingredients:
        # Top co-occurring ingredients = this ingredient's "role" in recipes
        top_neighbors = set(
            n for n, _ in cooccur[ing].most_common(50)
        )
        context[ing] = top_neighbors

    # Score pairs by: high context overlap + low direct co-occurrence
    substitution_scores = []
    common_list = sorted(common_ingredients)

    for i, a in enumerate(common_list):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(common_list)} ingredients...")

        for b in common_list[i + 1:]:
            ctx_a = context.get(a, set())
            ctx_b = context.get(b, set())

            if not ctx_a or not ctx_b:
                continue

            # Jaccard similarity of contexts (excluding each other)
            ctx_a_clean = ctx_a - {b}
            ctx_b_clean = ctx_b - {a}
            if not ctx_a_clean or not ctx_b_clean:
                continue

            jaccard = len(ctx_a_clean & ctx_b_clean) / len(ctx_a_clean | ctx_b_clean)

            # Direct co-occurrence rate (should be LOW for substitutes)
            direct_cooccur = cooccur[a].get(b, 0)
            max_possible = min(ingredient_recipes[a], ingredient_recipes[b])
            if max_possible == 0:
                continue
            cooccur_rate = direct_cooccur / max_possible

            # Substitution score = high context similarity + low co-occurrence
            # Good substitutes: jaccard > 0.3, cooccur_rate < 0.2
            if jaccard > 0.2 and cooccur_rate < 0.3:
                score = jaccard * (1 - cooccur_rate)
                substitution_scores.append((a, b, score, jaccard, cooccur_rate))

    # Sort by score
    substitution_scores.sort(key=lambda x: -x[2])

    # Build substitution map (top_k substitutes per ingredient)
    sub_map = defaultdict(list)
    for a, b, score, jaccard, cooccur_rate in substitution_scores[:5000]:
        if len(sub_map[a]) < top_k:
            sub_map[a].append({"substitute": b, "score": round(score, 4),
                               "context_similarity": round(jaccard, 4),
                               "cooccurrence_rate": round(cooccur_rate, 4)})
        if len(sub_map[b]) < top_k:
            sub_map[b].append({"substitute": a, "score": round(score, 4),
                               "context_similarity": round(jaccard, 4),
                               "cooccurrence_rate": round(cooccur_rate, 4)})

    # Save
    with open("mined_substitutions.json", "w") as f:
        json.dump(dict(sub_map), f, indent=2)

    # Also save a simple format for direct use in csv_to_triples_vitalbites.py
    simple_subs = {}
    for ing, subs in sub_map.items():
        simple_subs[ing] = [s["substitute"] for s in subs]

    with open("mined_substitutions_simple.json", "w") as f:
        json.dump(simple_subs, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SUBSTITUTION MINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Substitution pairs found:  {len(substitution_scores)}")
    print(f"  Ingredients with subs:     {len(sub_map)}")
    print(f"\n  Top 30 substitution pairs:")
    for a, b, score, jacc, cooc in substitution_scores[:30]:
        print(f"    {a:25s} ↔ {b:25s}  score={score:.3f}  ctx_sim={jacc:.3f}  cooccur={cooc:.3f}")

    print(f"\n  Saved to mined_substitutions.json (detailed)")
    print(f"  Saved to mined_substitutions_simple.json (for csv_to_triples_vitalbites.py)")

    return sub_map


def validate_substitutions(api_key: str, batch_size: int = 50):
    """
    Use Claude to validate and clean mined substitutions.

    The co-occurrence algorithm finds ingredients with similar recipe contexts
    but sometimes picks up CO-INGREDIENTS (onions with ground beef) instead of
    true SUBSTITUTES (ground turkey for ground beef).

    Claude filters these by asking: "Can ingredient B replace ingredient A
    in a recipe and serve the same culinary PURPOSE?"
    """
    import anthropic

    print("\n" + "=" * 70)
    print("STEP 3b: Validating substitutions with Claude")
    print("=" * 70)

    with open("mined_substitutions_simple.json") as f:
        raw_subs = json.load(f)

    print(f"  {len(raw_subs)} ingredients with candidate substitutes")
    total_candidates = sum(len(v) for v in raw_subs.values())
    print(f"  {total_candidates} total candidate pairs to validate")

    client = anthropic.Anthropic(api_key=api_key)

    # Batch the substitutions for validation
    items = list(raw_subs.items())
    validated_subs = {}
    removed_count = 0
    kept_count = 0

    num_batches = (len(items) + batch_size - 1) // batch_size
    print(f"  Sending {num_batches} batches of {batch_size} ingredients...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(items))
        batch = dict(items[start:end])

        prompt = f"""I have a recipe ingredient substitution map generated by a co-occurrence algorithm.
Some entries are WRONG — they list co-ingredients instead of true substitutes.

A TRUE SUBSTITUTE means ingredient B can REPLACE ingredient A in a recipe and serve 
the same culinary role/purpose. For example:
  - butter → olive oil ✓ (both are fats used for cooking)
  - ground beef → ground turkey ✓ (both are ground meat proteins)
  - ground beef → onions ✗ (onions are a co-ingredient, not a substitute)
  - sugar → honey ✓ (both are sweeteners)
  - chicken breast → salt ✗ (completely different roles)

For each ingredient, return ONLY the valid substitutes. Remove anything that is:
- A common co-ingredient rather than a replacement
- A completely different food category
- A seasoning/spice being listed as a substitute for a protein (or vice versa)
- An ingredient that serves a fundamentally different culinary purpose

Input:
{json.dumps(batch, indent=2)}

Return ONLY a JSON object with the same keys, but with the substitute lists 
cleaned to contain ONLY valid substitutions. If ALL candidates for an ingredient 
are invalid, return an empty list for that ingredient.

Return ONLY valid JSON, no other text."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            batch_validated = parse_json_response(response.content[0].text)

            for ing, subs in batch_validated.items():
                original_count = len(batch.get(ing, []))
                validated_subs[ing] = subs
                kept_count += len(subs)
                removed_count += original_count - len(subs)

            print(f"    Batch {batch_idx + 1}/{num_batches}: validated {len(batch_validated)} ingredients")

        except Exception as e:
            print(f"    Batch {batch_idx + 1} FAILED: {e}")
            # Keep original on failure
            for ing, subs in batch.items():
                validated_subs[ing] = subs
                kept_count += len(subs)

    # Remove empty entries
    validated_subs = {k: v for k, v in validated_subs.items() if v}

    with open("mined_substitutions_simple.json", "w") as f:
        json.dump(validated_subs, f, indent=2)

    # Also update the detailed version
    with open("mined_substitutions_validated.json", "w") as f:
        json.dump(validated_subs, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SUBSTITUTION VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Kept:      {kept_count} valid substitution pairs")
    print(f"  Removed:   {removed_count} false positives (co-ingredients)")
    print(f"  Accuracy:  {100 * kept_count / max(kept_count + removed_count, 1):.1f}% of candidates were valid")
    print(f"  Ingredients with valid subs: {len(validated_subs)}")
    print(f"\n  Saved to mined_substitutions_simple.json (overwritten)")
    print(f"  Saved to mined_substitutions_validated.json")

    return validated_subs


# ══════════════════════════════════════════════════════════════════
#  PART 4: NORMALIZE INGREDIENTS, NUTRIENTS, AND SUBSTITUTIONS
# ══════════════════════════════════════════════════════════════════

def normalize_entities(api_key: str, batch_size: int = 200):
    """
    Use Claude to normalize all entity names to canonical forms.

    The problem:
      Your 312K recipes have ~278K unique entities. Many are duplicates:
        "boneless chicken"     →  chicken_breast
        "chicken breast"       →  chicken_breast
        "boneless skinless chicken breast" → chicken_breast
        "longgrain rice"       →  long_grain_rice
        "long grain rice"      →  long_grain_rice
        "fresh lemon juice"    →  lemon_juice
        "lemon juice"          →  lemon_juice

    Without normalization, RotatE sees these as SEPARATE entities
    with no connection — it can't learn that Biryani and a stir-fry
    both use chicken if one says "boneless chicken" and the other
    says "chicken breast".

    Approach:
      1. Take the top N ingredients by frequency (covers 95%+ of triples)
      2. Send batches to Claude for canonical name assignment
      3. Apply rule-based normalization for the long tail
      4. Normalize nutrient names from the mined_nutrient_map
      5. Re-normalize substitution maps to use canonical names

    Output:
      mined_ingredient_normalize.json  — {raw_name: canonical_name}
      mined_nutrient_normalize.json    — {raw_name: canonical_name}
    """
    import anthropic

    print("\n" + "=" * 70)
    print("STEP 4: Normalizing entity names")
    print("=" * 70)

    # Load extracted data
    with open("extracted_texts.json") as f:
        extracted = json.load(f)

    ingredient_counts = extracted["ingredient_counts"]  # top 5000 by freq
    client = anthropic.Anthropic(api_key=api_key)

    # ──────────────────────────────────────────────────────────────
    # PHASE 1: Rule-based pre-normalization (free, no API cost)
    # ──────────────────────────────────────────────────────────────
    print("\n  Phase 1: Rule-based pre-normalization...")

    def rule_normalize(name: str) -> str:
        """Cheap deterministic normalization before hitting the API."""
        n = name.lower().strip()
        # Remove leading/trailing whitespace and punctuation
        n = re.sub(r"[,;.!]+$", "", n)
        # Remove "fresh", "dried", "ground", "chopped", etc. prefixes
        # that don't change what the ingredient IS
        prep_words = [
            "fresh", "dried", "ground", "chopped", "minced", "diced",
            "sliced", "crushed", "grated", "shredded", "frozen", "canned",
            "cooked", "raw", "organic", "large", "medium", "small",
            "whole", "halved", "quartered", "thinly", "finely", "roughly",
            "peeled", "deseeded", "seeded", "pitted", "boneless", "skinless",
            "unsalted", "salted", "roasted", "toasted", "melted", "softened",
            "plain", "pure", "extra-virgin", "extra virgin", "virgin",
            "low-fat", "low fat", "nonfat", "non-fat", "fat-free",
            "all-purpose", "all purpose", "unbleached", "sifted",
            "firmly packed", "lightly packed", "loosely packed",
        ]
        for word in prep_words:
            n = re.sub(rf"\b{re.escape(word)}\b", "", n)

        # Remove quantity artifacts
        n = re.sub(r"^\d+[\s/½¼¾⅓⅔⅛]*\s*", "", n)
        # Remove "of" artifacts: "zest of" → "zest", "juice of" → "juice"
        n = re.sub(r"\s+of\b", "", n)
        # Collapse whitespace
        n = re.sub(r"\s+", " ", n).strip()
        # Remove trailing 's' for simple plurals (but not "bass", "hummus", etc.)
        if len(n) > 4 and n.endswith("s") and not n.endswith(("ss", "us", "is")):
            singular = n[:-1]
            # Only de-pluralize if it doesn't make it too short
            if len(singular) > 2:
                n = singular
        # Standardize separators
        n = n.replace("-", " ").replace("  ", " ").strip()

        return n

    pre_normalized = {}
    for raw_name in ingredient_counts:
        pre_normalized[raw_name] = rule_normalize(raw_name)

    # Group by pre-normalized form to find clusters
    clusters = defaultdict(list)
    for raw, normed in pre_normalized.items():
        clusters[normed].append(raw)

    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    print(f"    {len(ingredient_counts)} raw ingredients")
    print(f"    {len(clusters)} after rule-based normalization")
    print(f"    {len(multi_clusters)} clusters with 2+ variants merged")
    print(f"    Examples of merged clusters:")
    for normed, raws in list(multi_clusters.items())[:10]:
        print(f"      '{normed}' ← {raws}")

    # ──────────────────────────────────────────────────────────────
    # PHASE 2: Claude normalizes remaining ambiguous ingredients
    # ──────────────────────────────────────────────────────────────
    print(f"\n  Phase 2: Claude API normalization (batched)...")

    # Get unique pre-normalized names, sorted by total frequency
    unique_prenormed = {}
    for normed, raws in clusters.items():
        total_freq = sum(ingredient_counts.get(r, 0) for r in raws)
        unique_prenormed[normed] = total_freq

    # Sort by frequency, take top ingredients for API normalization
    sorted_ingredients = sorted(unique_prenormed.items(), key=lambda x: -x[1])

    # We batch these to Claude in groups
    all_to_normalize = [name for name, _ in sorted_ingredients]
    canonical_map = {}  # pre_normalized_name → canonical_name

    num_batches = (len(all_to_normalize) + batch_size - 1) // batch_size
    print(f"    {len(all_to_normalize)} unique pre-normalized names")
    print(f"    Sending in {num_batches} batches of {batch_size}...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_to_normalize))
        batch = all_to_normalize[start:end]

        prompt = f"""I'm building a recipe knowledge graph. I need to normalize ingredient names 
to CANONICAL forms so that duplicates merge into single entities.

Rules for canonical names:
- Lowercase, underscores instead of spaces: "chicken breast" → "chicken_breast"
- Remove preparation methods: "diced tomatoes" → "tomato"
- Singularize: "onions" → "onion", "eggs" → "egg" 
- Keep the CORE ingredient identity: "baby spinach" → "spinach", but "baby corn" → "baby_corn" (different thing)
- Merge obvious duplicates: "cheddar" and "cheddar cheese" → "cheddar_cheese"
- Keep specificity when it matters: "dark chocolate" ≠ "milk chocolate", "red onion" ≠ "yellow onion"
- Common abbreviations: "evoo" → "olive_oil"
- "X rind" or "X zest" should stay as "X_zest" — they're distinct from the fruit itself
- "hot green chili pepper" → "green_chili_pepper"
- Compound ingredients stay compound: "soy sauce" → "soy_sauce", "baking powder" → "baking_powder"

Return ONLY a JSON object mapping each input name to its canonical form.
If a name is already canonical, map it to itself (with underscores).

Batch {batch_idx + 1}/{num_batches}:
{json.dumps(batch, indent=2)}

Return ONLY valid JSON, no other text."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            batch_map = parse_json_response(response.content[0].text)
            canonical_map.update(batch_map)
            print(f"    Batch {batch_idx + 1}/{num_batches}: "
                  f"mapped {len(batch_map)} ingredients")
        except Exception as e:
            print(f"    Batch {batch_idx + 1} FAILED: {e}")
            # Fallback: use pre-normalized names with underscores
            for name in batch:
                canonical_map[name] = name.replace(" ", "_")

    # ──────────────────────────────────────────────────────────────
    # PHASE 3: Build complete raw → canonical mapping
    # ──────────────────────────────────────────────────────────────
    print(f"\n  Phase 3: Building complete normalization map...")

    # Chain: raw_name → pre_normalized → canonical
    full_ingredient_map = {}
    for raw_name in ingredient_counts:
        pre_normed = pre_normalized[raw_name]
        canonical = canonical_map.get(pre_normed, pre_normed.replace(" ", "_"))
        full_ingredient_map[raw_name] = canonical

    # Stats
    unique_raw = len(full_ingredient_map)
    unique_canonical = len(set(full_ingredient_map.values()))
    reduction = unique_raw - unique_canonical

    print(f"    {unique_raw} raw ingredient names")
    print(f"    {unique_canonical} canonical names")
    print(f"    {reduction} duplicates eliminated ({100*reduction/max(unique_raw,1):.1f}% reduction)")

    # Show biggest merges
    reverse_map = defaultdict(list)
    for raw, canonical in full_ingredient_map.items():
        reverse_map[canonical].append(raw)

    big_merges = sorted(reverse_map.items(), key=lambda x: -len(x[1]))
    print(f"\n    Top 20 merged groups:")
    for canonical, raws in big_merges[:20]:
        if len(raws) > 1:
            freq = sum(ingredient_counts.get(r, 0) for r in raws)
            print(f"      {canonical:30s} ← {raws} (total freq: {freq:,})")

    with open("mined_ingredient_normalize.json", "w") as f:
        json.dump(full_ingredient_map, f, indent=2)

    # ──────────────────────────────────────────────────────────────
    # PHASE 4: Normalize nutrient names
    # ──────────────────────────────────────────────────────────────
    print(f"\n  Phase 4: Normalizing nutrient names...")

    # Load the mined nutrient map if it exists
    nutrient_normalize = {}
    try:
        with open("mined_nutrient_map.json") as f:
            nutrient_map = json.load(f)

        # Get all unique nutrient names from the map
        raw_nutrients = sorted(set(nutrient_map.values()))

        prompt_nutrients = f"""Normalize these micronutrient names to standard canonical forms.

Rules:
- Use lowercase with underscores
- Use common scientific/nutritional names
- Merge duplicates: "vitamin_b1" and "thiamin" → "thiamin_b1"
- Keep the most recognized name as primary
- Standard forms: iron, zinc, calcium, magnesium, potassium, sodium, phosphorus,
  selenium, manganese, iodine, copper, chromium, molybdenum, fluoride,
  vitamin_a, vitamin_b1_thiamin, vitamin_b2_riboflavin, vitamin_b3_niacin,
  vitamin_b5_pantothenic_acid, vitamin_b6, vitamin_b7_biotin, vitamin_b9_folate,
  vitamin_b12, vitamin_c, vitamin_d, vitamin_e, vitamin_k1, vitamin_k2,
  omega_3, omega_6, choline

Return a JSON object mapping each input name to its canonical form.

Input nutrients:
{json.dumps(raw_nutrients, indent=2)}

Return ONLY valid JSON, no other text."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt_nutrients}],
        )
        nutrient_normalize = parse_json_response(response.content[0].text)
        print(f"    Normalized {len(nutrient_normalize)} nutrient names")

    except FileNotFoundError:
        print("    mined_nutrient_map.json not found — run 'enrich' first")
    except Exception as e:
        print(f"    Nutrient normalization failed: {e}")

    with open("mined_nutrient_normalize.json", "w") as f:
        json.dump(nutrient_normalize, f, indent=2)

    # ──────────────────────────────────────────────────────────────
    # PHASE 5: Re-normalize substitutions to use canonical names
    # ──────────────────────────────────────────────────────────────
    print(f"\n  Phase 5: Re-normalizing substitutions with canonical names...")

    try:
        with open("mined_substitutions_simple.json") as f:
            raw_subs = json.load(f)

        normalized_subs = {}
        for raw_ing, raw_sub_list in raw_subs.items():
            canonical_ing = full_ingredient_map.get(raw_ing, raw_ing.replace(" ", "_"))
            canonical_sub_list = []
            for raw_sub in raw_sub_list:
                canonical_sub = full_ingredient_map.get(raw_sub, raw_sub.replace(" ", "_"))
                if canonical_sub != canonical_ing:  # don't substitute with yourself
                    canonical_sub_list.append(canonical_sub)

            if canonical_sub_list:
                # Merge if canonical_ing already exists
                if canonical_ing in normalized_subs:
                    existing = set(normalized_subs[canonical_ing])
                    existing.update(canonical_sub_list)
                    normalized_subs[canonical_ing] = sorted(existing)
                else:
                    normalized_subs[canonical_ing] = sorted(set(canonical_sub_list))

        with open("mined_substitutions_normalized.json", "w") as f:
            json.dump(normalized_subs, f, indent=2)

        print(f"    {len(raw_subs)} raw substitution groups → {len(normalized_subs)} canonical groups")

    except FileNotFoundError:
        print("    mined_substitutions_simple.json not found — run 'substitutions' first")

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"NORMALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Ingredient normalization: {unique_raw} → {unique_canonical} entities ({reduction} merged)")
    print(f"  Nutrient normalization:   {len(nutrient_normalize)} standardized")
    print(f"\n  Output files:")
    print(f"    mined_ingredient_normalize.json  — raw ingredient → canonical")
    print(f"    mined_nutrient_normalize.json    — raw nutrient → canonical")
    print(f"    mined_substitutions_normalized.json — substitutions with canonical names")
    print(f"\n  IMPACT ON KNOWLEDGE GRAPH:")
    print(f"    Before: {unique_raw} ingredient entities (many duplicates)")
    print(f"    After:  {unique_canonical} ingredient entities (clean, merged)")
    print(f"    This means RotatE has fewer, denser entities to learn from")
    print(f"    → better embeddings → better recommendations")

    return full_ingredient_map, nutrient_normalize


# ══════════════════════════════════════════════════════════════════
#  PART 5: GENERATE UPDATED csv_to_triples_vitalbites.py DICTS
# ══════════════════════════════════════════════════════════════════

def generate_updated_config():
    """
    Combine all mined knowledge into a single config file that can be
    imported by csv_to_triples_vitalbites.py.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Generating updated configuration")
    print("=" * 70)

    output = {}

    # Load mined files
    for fname, key in [
        ("mined_nutrient_map.json", "MICRO_DESCRIPTION_TO_NUTRIENT"),
        ("mined_health_labels.json", "HEALTH_DESCRIPTION_TO_LABEL"),
        ("mined_ailments.json", "AILMENT_NUTRIENT_MAP"),
        ("mined_substitutions_normalized.json", "SUBSTITUTIONS"),
        ("mined_substitutions_simple.json", "SUBSTITUTIONS_RAW"),
        ("mined_ingredient_normalize.json", "INGREDIENT_NORMALIZE"),
        ("mined_nutrient_normalize.json", "NUTRIENT_NORMALIZE"),
    ]:
        try:
            with open(fname) as f:
                output[key] = json.load(f)
            print(f"  Loaded {fname}: {len(output[key])} entries")
        except FileNotFoundError:
            print(f"  WARNING: {fname} not found, skipping")

    with open("mined_config.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print stats
    print(f"\n  Combined config saved to mined_config.json")
    if "AILMENT_NUTRIENT_MAP" in output:
        ailments = output["AILMENT_NUTRIENT_MAP"]
        total_needs = sum(len(v.get("needs", [])) for v in ailments.values())
        total_avoids = sum(len(v.get("avoid", [])) for v in ailments.values())
        print(f"\n  AILMENT COVERAGE:")
        print(f"    Conditions:        {len(ailments)}")
        print(f"    BENEFITS_FROM:     {total_needs} edges")
        print(f"    SHOULD_AVOID:      {total_avoids} edges")
        print(f"    (was 14 conditions / 59 edges in hand-curated version)")

    if "SUBSTITUTIONS" in output:
        subs = output["SUBSTITUTIONS"]
        total_pairs = sum(len(v) for v in subs.values())
        print(f"\n  SUBSTITUTION COVERAGE:")
        print(f"    Ingredients with substitutes: {len(subs)}")
        print(f"    Total substitute pairs:       {total_pairs}")
        print(f"    (was 34 ingredients / 172 edges in hand-curated version)")

    print(f"\n  Next step:")
    print(f"    Copy the dictionaries from mined_config.json into")
    print(f"    csv_to_triples_vitalbites.py, replacing the hand-curated ones.")
    print(f"    Then re-run: python csv_to_triples_vitalbites.py --input df_foodcom_recipes_filtered.csv")
    print(f"\n  Or use the auto-apply approach:")
    print(f"    python csv_to_triples_vitalbites.py --input df_foodcom_recipes_filtered.csv --config mined_config.json")

    if "INGREDIENT_NORMALIZE" in output:
        norm = output["INGREDIENT_NORMALIZE"]
        unique_canonical = len(set(norm.values()))
        print(f"\n  INGREDIENT NORMALIZATION:")
        print(f"    Raw names:        {len(norm)}")
        print(f"    Canonical names:  {unique_canonical}")
        print(f"    Duplicates merged: {len(norm) - unique_canonical}")

    if "NUTRIENT_NORMALIZE" in output:
        nn = output["NUTRIENT_NORMALIZE"]
        print(f"\n  NUTRIENT NORMALIZATION:")
        print(f"    Raw names:        {len(nn)}")
        print(f"    Canonical names:  {len(set(nn.values()))}")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VitalBites Knowledge Miner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step by step:
  python mine_knowledge.py extract --input df_foodcom_recipes_filtered.csv
  python mine_knowledge.py enrich --api-key sk-ant-api03-...
  python mine_knowledge.py substitutions --input df_foodcom_recipes_filtered.csv
  python mine_knowledge.py normalize --api-key sk-ant-api03-...
  python mine_knowledge.py config

  # All at once (recommended):
  python mine_knowledge.py all --input df_foodcom_recipes_filtered.csv --api-key sk-ant-api03-...
        """,
    )
    parser.add_argument("command", choices=["extract", "enrich", "normalize", "substitutions", "validate_subs", "config", "all"])
    parser.add_argument("--input", default="df_foodcom_recipes_filtered.csv")
    parser.add_argument("--api-key", help="Anthropic API key for Claude enrichment")
    parser.add_argument("--min-recipes", type=int, default=50, help="Min recipes for substitution mining")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for normalization API calls")
    args = parser.parse_args()

    if args.command == "extract":
        extract_unique_texts(args.input)

    elif args.command == "enrich":
        if not args.api_key:
            print("ERROR: --api-key required for enrichment")
            sys.exit(1)
        enrich_with_claude(args.api_key)

    elif args.command == "normalize":
        if not args.api_key:
            print("ERROR: --api-key required for normalization")
            sys.exit(1)
        normalize_entities(args.api_key, batch_size=args.batch_size)

    elif args.command == "substitutions":
        mine_substitutions(args.input, min_recipes=args.min_recipes)

    elif args.command == "validate_subs":
        if not args.api_key:
            print("ERROR: --api-key required for substitution validation")
            sys.exit(1)
        validate_substitutions(args.api_key)

    elif args.command == "config":
        generate_updated_config()

    elif args.command == "all":
        if not args.api_key:
            print("ERROR: --api-key required for 'all' command")
            sys.exit(1)
        extract_unique_texts(args.input)
        enrich_with_claude(args.api_key)
        mine_substitutions(args.input, min_recipes=args.min_recipes)
        validate_substitutions(args.api_key)
        normalize_entities(args.api_key, batch_size=args.batch_size)
        generate_updated_config()
