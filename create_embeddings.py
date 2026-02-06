#!/usr/bin/env python3
"""
Recipe Embedding Generator with Ingredient Substitution Awareness
=================================================================
Generates embeddings for recipes based on their ingredients, treating
common substitutes (e.g., olive oil ↔ butter, ghee ↔ butter) as similar.

The approach:
1. Parse all ingredients from the CSV
2. Build a substitution graph by scanning the corpus for known substitute pairs
3. Create a unified ingredient vocabulary with substitute-group IDs
4. Encode each recipe as a TF-IDF vector over substitute-aware ingredient groups
5. Reduce dimensionality with SVD to produce dense embeddings
6. Output: CSV of recipe names + embedding dimensions

Requirements:  pip install pandas scikit-learn numpy
"""

import re
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ──────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
INPUT_CSV = ".df_foodcom_recipes_filtered.csv"
OUTPUT_CSV = "recipe_embeddings.csv"
EMBEDDING_DIM = 64  # Number of embedding dimensions (tune as needed)

# ──────────────────────────────────────────────────────────────────────
# 2.  KNOWN INGREDIENT SUBSTITUTION PAIRS
#     Each tuple is (ingredient_a, ingredient_b) meaning a ↔ b.
#     These are the "seed" pairs. The script also auto-discovers
#     additional substitutes from the ingredient corpus (step 4).
# ──────────────────────────────────────────────────────────────────────
SUBSTITUTION_PAIRS = [
    # Fats & oils
    ("butter", "olive oil"),
    ("butter", "margarine"),
    ("butter", "coconut oil"),
    ("butter", "ghee"),
    ("vegetable oil", "canola oil"),
    ("vegetable oil", "olive oil"),
    ("vegetable oil", "sunflower oil"),
    ("vegetable oil", "corn oil"),
    ("vegetable oil", "peanut oil"),
    ("olive oil", "avocado oil"),
    ("shortening", "butter"),
    ("shortening", "coconut oil"),
    ("ghee", "clarified butter"),
    ("cooking spray", "vegetable oil"),
    ("cooking spray", "olive oil"),

    # Sugars & sweeteners
    ("sugar", "granulated sugar"),
    ("sugar", "honey"),
    ("sugar", "maple syrup"),
    ("sugar", "agave nectar"),
    ("brown sugar", "sugar"),
    ("brown sugar", "coconut sugar"),
    ("powdered sugar", "confectioners sugar"),
    ("powdered sugar", "icing sugar"),
    ("corn syrup", "honey"),
    ("corn syrup", "maple syrup"),
    ("molasses", "dark corn syrup"),

    # Dairy milks
    ("milk", "whole milk"),
    ("milk", "skim milk"),
    ("milk", "2% milk"),
    ("milk", "almond milk"),
    ("milk", "soy milk"),
    ("milk", "oat milk"),
    ("milk", "coconut milk"),
    ("buttermilk", "milk"),
    ("evaporated milk", "heavy cream"),
    ("half and half", "milk"),
    ("half and half", "heavy cream"),

    # Creams
    ("heavy cream", "whipping cream"),
    ("heavy cream", "heavy whipping cream"),
    ("heavy cream", "coconut cream"),
    ("sour cream", "greek yogurt"),
    ("sour cream", "plain yogurt"),
    ("cream cheese", "mascarpone"),
    ("cream cheese", "neufchatel cheese"),

    # Yogurts
    ("plain yogurt", "greek yogurt"),
    ("plain yogurt", "vanilla yogurt"),
    ("yogurt", "plain yogurt"),
    ("yogurt", "greek yogurt"),

    # Cheeses
    ("parmesan cheese", "pecorino romano"),
    ("parmesan cheese", "asiago cheese"),
    ("cheddar cheese", "colby cheese"),
    ("cheddar cheese", "monterey jack cheese"),
    ("mozzarella cheese", "provolone cheese"),
    ("ricotta cheese", "cottage cheese"),
    ("gruyere cheese", "swiss cheese"),
    ("feta cheese", "goat cheese"),

    # Eggs
    ("eggs", "egg"),
    ("egg whites", "eggs"),
    ("egg substitute", "eggs"),

    # Flours & starches
    ("all-purpose flour", "flour"),
    ("all-purpose flour", "bread flour"),
    ("all-purpose flour", "whole wheat flour"),
    ("cake flour", "all-purpose flour"),
    ("self-rising flour", "all-purpose flour"),
    ("cornstarch", "arrowroot"),
    ("cornstarch", "tapioca starch"),
    ("cornstarch", "potato starch"),
    ("almond flour", "coconut flour"),

    # Leavening
    ("baking powder", "baking soda"),

    # Vinegars
    ("white vinegar", "apple cider vinegar"),
    ("white vinegar", "red wine vinegar"),
    ("white vinegar", "rice vinegar"),
    ("balsamic vinegar", "red wine vinegar"),
    ("lemon juice", "lime juice"),
    ("lemon juice", "fresh lemon juice"),
    ("lime juice", "fresh lime juice"),
    ("fresh lemon juice", "lemon juice"),

    # Soy & Asian sauces
    ("soy sauce", "tamari"),
    ("soy sauce", "coconut aminos"),
    ("fish sauce", "soy sauce"),
    ("rice wine", "mirin"),
    ("rice vinegar", "rice wine vinegar"),
    ("hoisin sauce", "oyster sauce"),
    ("sesame oil", "toasted sesame oil"),

    # Mustards & condiments
    ("dijon mustard", "yellow mustard"),
    ("dijon mustard", "whole grain mustard"),
    ("ketchup", "tomato paste"),
    ("mayonnaise", "greek yogurt"),
    ("hot sauce", "cayenne pepper"),
    ("worcestershire sauce", "soy sauce"),

    # Tomato products
    ("tomato sauce", "tomato paste"),
    ("tomato sauce", "crushed tomatoes"),
    ("diced tomatoes", "crushed tomatoes"),
    ("diced tomatoes", "fresh tomatoes"),
    ("diced tomatoes", "tomatoes"),
    ("tomato paste", "tomato puree"),
    ("tomatoes", "fresh tomatoes"),
    ("tomatoes", "canned tomatoes"),

    # Alliums
    ("onion", "onions"),
    ("onion", "shallots"),
    ("onion", "leeks"),
    ("green onions", "scallions"),
    ("green onions", "chives"),
    ("garlic", "garlic powder"),
    ("garlic", "minced garlic"),
    ("onion powder", "onion"),

    # Herbs (fresh ↔ dried)
    ("fresh basil", "dried basil"),
    ("fresh oregano", "dried oregano"),
    ("fresh thyme", "dried thyme"),
    ("fresh rosemary", "dried rosemary"),
    ("fresh parsley", "dried parsley"),
    ("fresh dill", "dried dill"),
    ("fresh cilantro", "cilantro"),
    ("cilantro", "parsley"),
    ("mint leaf", "fresh mint"),
    ("mint leaf", "dried mint"),
    ("basil", "fresh basil"),
    ("oregano", "fresh oregano"),
    ("thyme", "fresh thyme"),

    # Spices
    ("cayenne pepper", "red pepper flakes"),
    ("cayenne pepper", "chili powder"),
    ("paprika", "smoked paprika"),
    ("ground cinnamon", "cinnamon"),
    ("ground ginger", "fresh ginger"),
    ("ground nutmeg", "nutmeg"),
    ("cumin", "cumin seed"),
    ("ground cumin", "cumin seed"),
    ("ground coriander", "coriander seed"),

    # Peppers
    ("hot green chili peppers", "jalapeno peppers"),
    ("hot green chili peppers", "serrano peppers"),
    ("jalapeno peppers", "serrano peppers"),
    ("bell pepper", "red bell pepper"),
    ("bell pepper", "green bell pepper"),
    ("red bell pepper", "green bell pepper"),
    ("chipotle peppers", "ancho chili"),

    # Proteins
    ("boneless chicken", "chicken breast"),
    ("boneless chicken", "chicken thighs"),
    ("chicken breast", "chicken thighs"),
    ("ground beef", "ground turkey"),
    ("ground beef", "ground pork"),
    ("ground turkey", "ground chicken"),
    ("bacon", "pancetta"),
    ("bacon", "turkey bacon"),
    ("ham", "prosciutto"),
    ("shrimp", "prawns"),
    ("salmon", "trout"),
    ("tuna", "swordfish"),
    ("tofu", "tempeh"),

    # Rice & grains
    ("basmati rice", "longgrain rice"),
    ("basmati rice", "jasmine rice"),
    ("longgrain rice", "jasmine rice"),
    ("white rice", "basmati rice"),
    ("white rice", "longgrain rice"),
    ("brown rice", "white rice"),
    ("quinoa", "couscous"),
    ("quinoa", "bulgur"),
    ("couscous", "orzo"),
    ("pasta", "spaghetti"),
    ("pasta", "penne"),
    ("pasta", "egg noodles"),

    # Nuts
    ("cashews", "almonds"),
    ("cashews", "peanuts"),
    ("walnuts", "pecans"),
    ("almonds", "pecans"),
    ("pine nuts", "walnuts"),
    ("peanut butter", "almond butter"),

    # Dried fruits
    ("raisins", "dried cranberries"),
    ("raisins", "currants"),
    ("dried apricots", "dried cherries"),

    # Broths & stocks
    ("chicken broth", "chicken stock"),
    ("chicken broth", "vegetable broth"),
    ("beef broth", "beef stock"),
    ("beef broth", "vegetable broth"),
    ("vegetable broth", "vegetable stock"),

    # Salts
    ("salt", "sea salt"),
    ("salt", "kosher salt"),

    # Breads & breadcrumbs
    ("breadcrumbs", "panko breadcrumbs"),
    ("bread", "tortillas"),
    ("croutons", "breadcrumbs"),

    # Beans & legumes
    ("black beans", "kidney beans"),
    ("black beans", "pinto beans"),
    ("kidney beans", "cannellini beans"),
    ("chickpeas", "white beans"),
    ("lentils", "split peas"),

    # Leafy greens
    ("spinach", "kale"),
    ("spinach", "swiss chard"),
    ("arugula", "watercress"),
    ("romaine lettuce", "iceberg lettuce"),

    # Potatoes
    ("russet potatoes", "yukon gold potatoes"),
    ("sweet potatoes", "yams"),
    ("potatoes", "russet potatoes"),

    # Citrus
    ("lemon", "lime"),
    ("lemon zest", "lime zest"),
    ("lemon zest of", "lemon zest"),
    ("orange juice", "tangerine juice"),
    ("orange zest", "lemon zest"),
]

# ──────────────────────────────────────────────────────────────────────
# 3.  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def parse_ingredients(raw: str) -> list[str]:
    """Parse the numpy-style string array from the CSV into a Python list."""
    if pd.isna(raw):
        return []
    items = re.findall(r"'([^']*?)'", raw)
    return [item.strip().lower() for item in items if item.strip()]


def build_substitution_groups(pairs: list[tuple[str, str]],
                              corpus_ingredients: set[str]) -> dict[str, str]:
    """
    Build a Union-Find over substitute pairs, then map every ingredient
    to a canonical group representative.

    Only ingredients that actually appear in the corpus are kept, so the
    vocabulary stays grounded in the data.

    Returns: dict mapping ingredient_name -> canonical_group_name
    """
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Prefer the shorter / more common name as canonical
            parent[rb] = ra

    # Union all substitute pairs
    for a, b in pairs:
        a_low, b_low = a.lower().strip(), b.lower().strip()
        union(a_low, b_low)

    # Build mapping for every ingredient in the corpus
    mapping = {}
    for ing in corpus_ingredients:
        canonical = find(ing)
        mapping[ing] = canonical

    return mapping


def auto_discover_substitutes(all_ingredients: set[str]) -> list[tuple[str, str]]:
    """
    Heuristically discover additional substitute pairs from the corpus
    by matching ingredient name patterns:
      - "fresh X" ↔ "X"
      - "dried X" ↔ "X"
      - "ground X" ↔ "X"
      - Plural ↔ singular (simple 's' suffix)
      - "X rind of" / "X zest of" variants
    """
    discovered = []
    ing_set = set(all_ingredients)

    for ing in all_ingredients:
        # fresh/dried/ground prefix patterns
        for prefix in ("fresh ", "dried ", "ground ", "chopped ", "minced ",
                       "crushed ", "diced ", "sliced ", "shredded "):
            if ing.startswith(prefix):
                base = ing[len(prefix):]
                if base in ing_set:
                    discovered.append((ing, base))

        # Simple plural: "onions" -> "onion"
        if ing.endswith("s") and ing[:-1] in ing_set:
            discovered.append((ing, ing[:-1]))

        # "X rind of" / "X zest of" -> "X"
        for suffix in (" rind of", " zest of", " juice"):
            if ing.endswith(suffix):
                base = ing[: -len(suffix)].strip()
                if base in ing_set:
                    discovered.append((ing, base))

    return discovered


# ──────────────────────────────────────────────────────────────────────
# 4.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ─────────────────────────────────────────────────────
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} recipes loaded.")

    # ── Parse ingredients ─────────────────────────────────────────────
    df["ingredient_list"] = df["cleaned_ingredients"].apply(parse_ingredients)

    all_ingredients: set[str] = set()
    for ing_list in df["ingredient_list"]:
        all_ingredients.update(ing_list)
    print(f"  {len(all_ingredients)} unique ingredients found in corpus.")

    # ── Build substitution groups ─────────────────────────────────────
    # Combine hand-curated pairs with auto-discovered ones
    auto_pairs = auto_discover_substitutes(all_ingredients)
    all_pairs = SUBSTITUTION_PAIRS + auto_pairs
    print(f"  {len(SUBSTITUTION_PAIRS)} curated substitute pairs + "
          f"{len(auto_pairs)} auto-discovered = {len(all_pairs)} total.")

    sub_map = build_substitution_groups(all_pairs, all_ingredients)

    # Show discovered substitution groups (for inspection)
    groups = defaultdict(set)
    for ing, canonical in sub_map.items():
        groups[canonical].add(ing)
    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  {len(multi_groups)} substitution groups with 2+ members:")
    for canonical, members in sorted(multi_groups.items()):
        print(f"    {canonical}: {sorted(members)}")

    # ── Map ingredients to canonical names ────────────────────────────
    def canonicalize(ingredient_list: list[str]) -> list[str]:
        """Replace each ingredient with its canonical substitute-group name."""
        return [sub_map.get(ing, ing) for ing in ingredient_list]

    df["canonical_ingredients"] = df["ingredient_list"].apply(canonicalize)

    # ── Build TF-IDF matrix ───────────────────────────────────────────
    # Join canonical ingredients into a "document" string per recipe.
    # We use ingredient names as pseudo-tokens.
    df["ingredient_doc"] = df["canonical_ingredients"].apply(
        lambda lst: " ".join(token.replace(" ", "_") for token in lst)
    )

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\s]+",  # each underscore-joined ingredient is one token
        max_features=5000,
        sublinear_tf=True,        # dampens high-frequency ingredients like "salt"
    )
    tfidf_matrix = vectorizer.fit_transform(df["ingredient_doc"])
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    # ── Reduce to dense embeddings via SVD ────────────────────────────
    n_components = min(EMBEDDING_DIM, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    if n_components < 1:
        print("  WARNING: Not enough data for SVD. Falling back to sparse TF-IDF.")
        embeddings = tfidf_matrix.toarray()
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)
        explained = svd.explained_variance_ratio_.sum()
        print(f"  SVD: {n_components} components, {explained:.1%} variance explained.")

    # L2-normalize so cosine similarity = dot product
    embeddings = normalize(embeddings, norm="l2")
    print(f"  Final embedding shape: {embeddings.shape}")

    # ── Build output DataFrame ────────────────────────────────────────
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
    out_df = pd.concat([df[["Name"]].reset_index(drop=True),
                        emb_df.reset_index(drop=True)], axis=1)

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved {len(out_df)} recipe embeddings ({embeddings.shape[1]}d) to {OUTPUT_CSV}")

    # ── Quick sanity check: show pairwise similarities ────────────────
    if len(out_df) > 1:
        sim_matrix = embeddings @ embeddings.T
        print("\nPairwise cosine similarities:")
        names = out_df["Name"].tolist()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"  {names[i]}  ↔  {names[j]}: {sim_matrix[i, j]:.4f}")


if __name__ == "__main__":
    main()
