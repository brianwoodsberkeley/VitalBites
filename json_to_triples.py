"""
JSON → Knowledge Graph Triples Extractor + RotatE Training Pipeline
====================================================================
Takes a recipe JSON with nested, multi-edge data and produces:
  1. triples.tsv — flat (head, relation, tail) file for PyKEEN
  2. Trained RotatE model with entity/relation embeddings

The key challenge: ONE recipe fans out into MANY triples because:
  - A recipe has multiple ingredients         (1 recipe → N ingredient edges)
  - Each ingredient has multiple nutrients     (1 ingredient → M nutrient edges)  
  - Each macronutrient has multiple micros     (1 macro → K micro edges)
  - Ingredients have substitution relationships (1 ingredient → S substitute edges)
  - Recipes map to health functions, categories, ailments...

So a single recipe JSON record can generate 20-40+ triples.

Usage:
    pip install pykeen pandas torch
    python json_to_triples.py --input sample_recipe_data.json --output triples.tsv
    python json_to_triples.py --input sample_recipe_data.json --train  # extract + train RotatE

Author: Brian / LeeroyChainkins AI
"""

import json
import csv
import argparse
from collections import Counter
from pathlib import Path


# ──────────────────────────────────────────────────────────────────
# STEP 1: DEFINE YOUR RELATION TYPES
# ──────────────────────────────────────────────────────────────────
# These are the edge types in your knowledge graph.
# RotatE will learn a separate rotation vector for each relation.
#
# RELATION                  HEAD TYPE       TAIL TYPE       PATTERN
# ─────────────────────────────────────────────────────────────────
# CONTAINS_INGREDIENT       Recipe          Ingredient      1-to-many
# IN_CATEGORY               Recipe          Category        many-to-1
# IN_SUBCATEGORY            Recipe          Subcategory     many-to-1
# HAS_SUBCATEGORY           Category        Subcategory     1-to-many
# HAS_HEALTH_FUNCTION       Recipe          HealthFunc      many-to-many
# HIGH_IN                   Recipe          MacroNutrient   many-to-many
# PROVIDES                  Recipe          MicroNutrient   many-to-many
# SUBSTITUTES_FOR           Ingredient      Ingredient      symmetric
# BENEFITS_FROM             Ailment         MicroNutrient   many-to-many
# SHOULD_AVOID              Ailment         MicroNutrient   many-to-many
# IS                        Recipe          Property        many-to-many
# ──────────────────────────────────────────────────────────────────


def clean_nutrient_name(nutrient_key: str) -> str:
    """Strip unit suffixes: 'iron_mg' → 'iron', 'vitamin_b12_mcg' → 'vitamin_b12'"""
    for suffix in ["_mg", "_mcg", "_iu", "_g"]:
        if nutrient_key.endswith(suffix):
            return nutrient_key[: -len(suffix)]
    return nutrient_key


def extract_triples(data: dict) -> list[tuple[str, str, str]]:
    """
    Extract (head, relation, tail) triples from the recipe JSON.
    
    This is where the multi-edge explosion happens:
    
    JSON structure:
        recipe
          ├── ingredients[]          → N CONTAINS_INGREDIENT edges
          ├── category               → 1 IN_CATEGORY edge
          ├── subcategory            → 1 IN_SUBCATEGORY edge
          ├── health_functions[]     → M HAS_HEALTH_FUNCTION edges
          ├── macronutrients{}       → K HIGH_IN edges (thresholded)
          └── micronutrients{}       → J PROVIDES edges (thresholded)
        
        substitutions{}              → S SUBSTITUTES_FOR edges (bidirectional)
        ailment_nutrient_map{}       → A BENEFITS_FROM / SHOULD_AVOID edges
    """
    triples = []

    # ── Micronutrient thresholds ──────────────────────────────────
    # Only create a PROVIDES edge if the recipe has a meaningful
    # amount of the nutrient. Tune these to your dataset.
    micro_thresholds = {
        "iron_mg": 2.0,
        "zinc_mg": 2.0,
        "calcium_mg": 150,
        "vitamin_b12_mcg": 1.0,
        "vitamin_d_iu": 100,
        "potassium_mg": 400,
        "folate_mcg": 80,
        "vitamin_c_mg": 15,
        "vitamin_k_mcg": 30,
        "omega3_g": 1.0,
        "selenium_mcg": 30,
        "magnesium_mg": 50,
        "vitamin_a_iu": 1000,
        "vitamin_e_mg": 2.0,
        "phosphorus_mg": 200,
    }

    # ── Recipe triples ────────────────────────────────────────────
    for recipe in data["recipes"]:
        name = recipe["name"]

        # Recipe → Ingredients (the biggest fan-out, 5-15 edges per recipe)
        for ing in recipe["ingredients"]:
            triples.append((name, "CONTAINS_INGREDIENT", ing["name"]))

        # Recipe → Category hierarchy
        triples.append((name, "IN_CATEGORY", recipe["category"]))
        triples.append((name, "IN_SUBCATEGORY", recipe["subcategory"]))
        triples.append(
            (recipe["category"], "HAS_SUBCATEGORY", recipe["subcategory"])
        )

        # Recipe → Health Functions (2-4 edges per recipe typically)
        for hf in recipe.get("health_functions", []):
            triples.append((name, "HAS_HEALTH_FUNCTION", hf))

        # Recipe → Macronutrient highlights (thresholded)
        macros = recipe.get("macronutrients", {})
        if macros.get("protein_g", 0) >= 20:
            triples.append((name, "HIGH_IN", "protein"))
        if macros.get("fiber_g", 0) >= 8:
            triples.append((name, "HIGH_IN", "fiber"))
        if macros.get("fat_g", 0) >= 30:
            triples.append((name, "HIGH_IN", "fat"))
        if macros.get("calories", 0) <= 300:
            triples.append((name, "IS", "low-calorie"))
        if macros.get("calories", 0) >= 500:
            triples.append((name, "IS", "high-calorie"))

        # Recipe → Micronutrients (thresholded — only meaningful amounts)
        for nutrient, value in recipe.get("micronutrients", {}).items():
            threshold = micro_thresholds.get(nutrient, 0)
            if value >= threshold:
                triples.append(
                    (name, "PROVIDES", clean_nutrient_name(nutrient))
                )

    # ── Substitution triples (bidirectional) ──────────────────────
    # RotatE CAN learn symmetric relations, but being explicit helps
    # with a small dataset. For 500K recipes you might only need one direction.
    for ingredient, subs in data.get("substitutions", {}).items():
        for sub in subs:
            triples.append((ingredient, "SUBSTITUTES_FOR", sub))
            triples.append((sub, "SUBSTITUTES_FOR", ingredient))

    # ── Ailment → Micronutrient triples ───────────────────────────
    for ailment, info in data.get("ailment_nutrient_map", {}).items():
        for nutrient in info.get("needs", []):
            triples.append(
                (ailment, "BENEFITS_FROM", clean_nutrient_name(nutrient))
            )
        for nutrient in info.get("avoid", []):
            triples.append(
                (ailment, "SHOULD_AVOID", clean_nutrient_name(nutrient))
            )

    # ── Deduplicate & sort ────────────────────────────────────────
    triples = sorted(set(triples), key=lambda t: (t[1], t[0], t[2]))

    return triples


def write_triples(triples: list[tuple], output_path: str):
    """Write triples to TSV (PyKEEN's expected format: head \\t relation \\t tail)"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for h, r, t in triples:
            writer.writerow([h, r, t])


def print_stats(triples: list[tuple]):
    """Print summary statistics about the extracted graph."""
    relation_counts = Counter(r for _, r, _ in triples)
    entities = set()
    for h, _, t in triples:
        entities.add(h)
        entities.add(t)

    print(f"\n{'='*60}")
    print(f"KNOWLEDGE GRAPH SUMMARY")
    print(f"{'='*60}")
    print(f"Total triples:    {len(triples)}")
    print(f"Unique entities:  {len(entities)}")
    print(f"Unique relations: {len(relation_counts)}")
    print(f"\nTriples per relation type:")
    for rel, count in relation_counts.most_common():
        print(f"  {rel:30s} {count:>5d}")
    print(f"\nSample triples:")
    for h, r, t in triples[:10]:
        print(f"  ({h})  --[{r}]-->  ({t})")
    print(f"  ...")


def train_rotate(triples_path: str, epochs: int = 200, embedding_dim: int = 128):
    """
    Train RotatE on the extracted triples using PyKEEN.
    
    This learns vector embeddings for every entity and relation
    in the complex vector space, where relations are rotations.
    """
    try:
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        print("\nERROR: PyKEEN not installed. Run:")
        print("  pip install pykeen")
        return None

    print(f"\n{'='*60}")
    print(f"TRAINING RotatE MODEL")
    print(f"{'='*60}")

    # Load triples
    tf = TriplesFactory.from_path(triples_path)

    # Split: 80% train, 10% validation, 10% test
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)

    print(f"Training triples:   {training.num_triples}")
    print(f"Validation triples: {validation.num_triples}")
    print(f"Test triples:       {testing.num_triples}")

    # Train RotatE
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model="RotatE",
        model_kwargs={
            "embedding_dim": embedding_dim,  # dimension of complex embeddings
        },
        training_kwargs={
            "num_epochs": epochs,
            "batch_size": 64,           # increase for larger datasets
        },
        training_loop="sLCWA",          # stochastic local closed world assumption
        negative_sampler="basic",
        negative_sampler_kwargs={
            "num_negs_per_pos": 32,     # negative samples per positive triple
        },
        optimizer="Adam",
        optimizer_kwargs={
            "lr": 0.001,
        },
        evaluator_kwargs={
            "filtered": True,           # filtered ranking (standard for KGE)
        },
        random_seed=42,
    )

    # Print evaluation metrics
    print(f"\nTest Results:")
    print(f"  MRR (Mean Reciprocal Rank): {result.metric_results.get_metric('mean_reciprocal_rank'):.4f}")
    print(f"  Hits@1:  {result.metric_results.get_metric('hits_at_1'):.4f}")
    print(f"  Hits@3:  {result.metric_results.get_metric('hits_at_3'):.4f}")
    print(f"  Hits@10: {result.metric_results.get_metric('hits_at_10'):.4f}")

    # Save the model
    result.save_to_directory("rotate_model")
    print(f"\nModel saved to ./rotate_model/")

    return result


def demo_recommendations(result, query_recipe: str = "Chicken Biryani"):
    """
    Demo: Use trained embeddings to find similar recipes and
    recommend recipes for a specific ailment.
    """
    import torch
    import numpy as np

    model = result.model
    training = result.training

    # Get entity-to-id mapping
    entity_to_id = training.entity_to_id

    if query_recipe not in entity_to_id:
        print(f"'{query_recipe}' not found in graph entities.")
        return

    print(f"\n{'='*60}")
    print(f"DEMO: Recommendations for '{query_recipe}'")
    print(f"{'='*60}")

    # Get all entity embeddings
    entity_embeddings = model.entity_representations[0](
        indices=torch.arange(training.num_entities)
    ).detach().numpy()

    # Find query recipe embedding
    query_id = entity_to_id[query_recipe]
    query_emb = entity_embeddings[query_id]

    # Compute distances to all entities
    distances = np.linalg.norm(entity_embeddings - query_emb, axis=1)

    # Get closest entities (filter to just recipes by checking names)
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    recipe_names = {r["name"] for r in json.load(open("sample_recipe_data.json"))["recipes"]}

    scored = []
    for eid in range(len(distances)):
        ename = id_to_entity[eid]
        if ename in recipe_names and ename != query_recipe:
            scored.append((ename, distances[eid]))

    scored.sort(key=lambda x: x[1])
    print(f"\nMost similar recipes to '{query_recipe}':")
    for name, dist in scored[:5]:
        print(f"  {name:30s}  distance={dist:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recipe JSON → KG Triples → RotatE")
    parser.add_argument("--input", default="sample_recipe_data.json", help="Input JSON file")
    parser.add_argument("--output", default="triples.tsv", help="Output triples TSV")
    parser.add_argument("--train", action="store_true", help="Also train RotatE model")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    args = parser.parse_args()

    # Load JSON
    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"  {len(data['recipes'])} recipes loaded")

    # Extract triples
    print("Extracting triples...")
    triples = extract_triples(data)
    print_stats(triples)

    # Write TSV
    write_triples(triples, args.output)
    print(f"\nTriples written to {args.output}")

    # Optionally train RotatE
    if args.train:
        result = train_rotate(args.output, epochs=args.epochs, embedding_dim=args.dim)
        if result:
            demo_recommendations(result)
