"""
RotatE Knowledge Graph: Training + Inference Pipeline
=====================================================
This script does two things:
  1. TRAIN  — Takes triples.tsv → trains RotatE → saves model + embeddings
  2. INFER  — Loads trained model → answers queries against the knowledge graph

The key insight: after training, every entity (recipe, ingredient, nutrient,
ailment) and every relation (CONTAINS_INGREDIENT, PROVIDES, BENEFITS_FROM, etc.)
lives in a complex vector space. Inference is just geometry — rotating and
measuring distances in that space.

Usage:
    # Step 1: Train (do this once, takes minutes on CPU, seconds on GPU)
    python train_and_infer.py train --triples triples.tsv --epochs 300 --dim 256

    # Step 2: Infer (do this as many times as you want)
    python train_and_infer.py similar   --recipe "Chicken Biryani" --top 10
    python train_and_infer.py recommend --ailment "anemia" --top 10
    python train_and_infer.py predict   --head "Chicken Biryani" --relation "PROVIDES" --top 10
    python train_and_infer.py explore   --entity "iron"

Requirements:
    pip install pykeen torch pandas numpy

Author: Brian / LeeroyChainkins AI
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ══════════════════════════════════════════════════════════════════
#  PART 1: TRAINING
# ══════════════════════════════════════════════════════════════════

def train(
    triples_path: str,
    output_dir: str = "trained_model",
    epochs: int = 300,
    embedding_dim: int = 256,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_negs: int = 64,
    seed: int = 42,
):
    """
    Train RotatE on the extracted triples.

    What happens during training:
    ─────────────────────────────
    1. Each entity gets a complex vector:  e ∈ ℂ^d
    2. Each relation gets a rotation vector: r ∈ ℂ^d  where |r_i| = 1
    3. For a true triple (h, r, t), we want:  h ∘ r ≈ t
       (element-wise rotation of head by relation should land near tail)
    4. Loss function pushes true triples to score high,
       random corrupted triples (negative samples) to score low
    5. After training, the geometry encodes all the knowledge:
       - Similar recipes cluster together
       - Ingredients that appear in similar contexts cluster
       - Relations capture typed transformations

    Hyperparameter guidance:
    ────────────────────────
    - embedding_dim: 128 for <10K triples, 256 for 10K-1M, 512 for >1M
    - epochs: 200-500 is typical. Watch validation MRR plateau.
    - num_negs: more = better but slower. 32-128 is the sweet spot.
    - lr: 1e-3 for Adam is usually fine. Reduce if loss oscillates.
    - batch_size: larger = faster but noisier. 256-1024 for most datasets.
    """
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    print("=" * 70)
    print("TRAINING RotatE KNOWLEDGE GRAPH EMBEDDINGS")
    print("=" * 70)

    # ── Load and split triples ────────────────────────────────────
    print(f"\nLoading triples from {triples_path}...")
    tf = TriplesFactory.from_path(triples_path)
    print(f"  Total triples:    {tf.num_triples}")
    print(f"  Unique entities:  {tf.num_entities}")
    print(f"  Unique relations: {tf.num_relations}")

    # 80/10/10 split — stratified by relation type
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=seed)
    print(f"  Training split:   {training.num_triples}")
    print(f"  Validation split: {validation.num_triples}")
    print(f"  Test split:       {testing.num_triples}")

    # ── Train RotatE ──────────────────────────────────────────────
    print(f"\nTraining RotatE (dim={embedding_dim}, epochs={epochs})...")
    print(f"  This may take a few minutes on CPU.\n")

    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,

        model="RotatE",
        model_kwargs={
            "embedding_dim": embedding_dim,
        },

        # Training loop: stochastic Local Closed World Assumption
        # (standard for KGE — assumes unobserved triples are false)
        training_loop="sLCWA",

        training_kwargs={
            "num_epochs": epochs,
            "batch_size": batch_size,
            "checkpoint_name": "checkpoint.pt",
            "checkpoint_frequency": 50,      # save every 50 epochs
            "checkpoint_directory": output_dir,
        },

        # Negative sampling: for each real triple, generate N fake ones
        # by corrupting either head or tail
        negative_sampler="basic",
        negative_sampler_kwargs={
            "num_negs_per_pos": num_negs,
        },

        optimizer="Adam",
        optimizer_kwargs={
            "lr": lr,
        },

        # Filtered evaluation: when ranking, remove other true triples
        # from candidates (standard practice, avoids penalizing correct answers)
        evaluator_kwargs={
            "filtered": True,
        },

        random_seed=seed,
    )

    # ── Print evaluation metrics ──────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    metrics = result.metric_results
    mrr = metrics.get_metric("both.realistic.inverse_harmonic_mean_rank")
    h1 = metrics.get_metric("both.realistic.hits_at_1")
    h3 = metrics.get_metric("both.realistic.hits_at_3")
    h10 = metrics.get_metric("both.realistic.hits_at_10")

    print(f"  MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"  Hits@1:                     {h1:.4f}")
    print(f"  Hits@3:                     {h3:.4f}")
    print(f"  Hits@10:                    {h10:.4f}")
    print()
    print("  Interpretation:")
    print("    MRR > 0.3  = good for most KG tasks")
    print("    MRR > 0.5  = very strong")
    print("    Hits@10    = % of time correct answer is in top 10 predictions")

    # ── Save model + artifacts ────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    result.save_to_directory(output_dir)

    # Also export entity/relation mappings as readable JSON
    entity_to_id = training.entity_to_id
    relation_to_id = training.relation_to_id

    with open(os.path.join(output_dir, "entity_to_id.json"), "w") as f:
        json.dump(entity_to_id, f, indent=2)

    with open(os.path.join(output_dir, "relation_to_id.json"), "w") as f:
        json.dump(relation_to_id, f, indent=2)

    # Export embeddings as numpy arrays for fast inference
    model = result.model
    entity_embs = model.entity_representations[0](
        indices=torch.arange(training.num_entities)
    ).detach().cpu().numpy()
    relation_embs = model.relation_representations[0](
        indices=torch.arange(training.num_relations)
    ).detach().cpu().numpy()

    np.save(os.path.join(output_dir, "entity_embeddings.npy"), entity_embs)
    np.save(os.path.join(output_dir, "relation_embeddings.npy"), relation_embs)

    print(f"\nModel and artifacts saved to ./{output_dir}/")
    print(f"  entity_embeddings.npy     shape: {entity_embs.shape}")
    print(f"  relation_embeddings.npy   shape: {relation_embs.shape}")
    print(f"  entity_to_id.json         {len(entity_to_id)} entities")
    print(f"  relation_to_id.json       {len(relation_to_id)} relations")

    return result


# ══════════════════════════════════════════════════════════════════
#  PART 2: INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════

class KnowledgeGraphInference:
    """
    Inference engine for a trained RotatE knowledge graph.

    This loads the saved embeddings and mappings and provides
    several query methods. No GPU needed for inference — it's
    just numpy vector math.

    How RotatE inference works:
    ──────────────────────────
    Given a query like (Chicken Biryani, PROVIDES, ???):
      1. Look up the head embedding:     h = entity_embs["Chicken Biryani"]
      2. Look up the relation embedding: r = relation_embs["PROVIDES"]
      3. Compute the predicted tail:     predicted_t = h ∘ r  (complex rotation)
      4. Score ALL entities by distance:  score(e) = ||h ∘ r - e||
      5. Rank by score (lower = better match)
      6. Return top-K entities

    For (???, BENEFITS_FROM, iron):
      Same idea but solve for the head: predicted_h = t ∘ conj(r)
    """

    def __init__(self, model_dir: str = "trained_model"):
        print(f"Loading trained model from {model_dir}...")

        # Load mappings
        with open(os.path.join(model_dir, "entity_to_id.json")) as f:
            self.entity_to_id = json.load(f)
        with open(os.path.join(model_dir, "relation_to_id.json")) as f:
            self.relation_to_id = json.load(f)

        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}

        # Load embeddings
        self.entity_embs = np.load(os.path.join(model_dir, "entity_embeddings.npy"))
        self.relation_embs = np.load(os.path.join(model_dir, "relation_embeddings.npy"))

        # RotatE uses complex embeddings stored as [real, imag] concatenated
        # Split them back into complex numbers for rotation math
        dim = self.entity_embs.shape[1] // 2
        self.entity_complex = (
            self.entity_embs[:, :dim] + 1j * self.entity_embs[:, dim:]
        )
        self.relation_complex = (
            self.relation_embs[:, :dim] + 1j * self.relation_embs[:, dim:]
        )

        print(f"  {len(self.entity_to_id)} entities, "
              f"{len(self.relation_to_id)} relations, "
              f"dim={dim}")

    def _score_all_tails(self, head_name: str, relation_name: str) -> np.ndarray:
        """
        Given (head, relation, ???), score every entity as a potential tail.
        Lower score = better match.

        Math: score(t) = ||h ∘ r - t||  (L1 norm in complex space)
        """
        h_id = self.entity_to_id[head_name]
        r_id = self.relation_to_id[relation_name]

        h = self.entity_complex[h_id]      # shape: (dim,)
        r = self.relation_complex[r_id]     # shape: (dim,)

        # Rotate head by relation
        predicted_tail = h * r              # element-wise complex multiplication

        # Distance to every entity
        diffs = self.entity_complex - predicted_tail  # shape: (num_entities, dim)
        scores = np.sum(np.abs(diffs), axis=1)        # L1 distance

        return scores

    def _score_all_heads(self, relation_name: str, tail_name: str) -> np.ndarray:
        """
        Given (???, relation, tail), score every entity as a potential head.

        Math: score(h) = ||h - t ∘ conj(r)||
              (since h ∘ r = t  →  h = t ∘ r⁻¹  →  h = t ∘ conj(r))
        """
        t_id = self.entity_to_id[tail_name]
        r_id = self.relation_to_id[relation_name]

        t = self.entity_complex[t_id]
        r = self.relation_complex[r_id]

        # Inverse rotation: multiply tail by conjugate of relation
        predicted_head = t * np.conj(r)

        diffs = self.entity_complex - predicted_head
        scores = np.sum(np.abs(diffs), axis=1)

        return scores

    # ──────────────────────────────────────────────────────────────
    #  QUERY 1: Link Prediction
    #  "What does Chicken Biryani PROVIDE?"
    #  "What CONTAINS_INGREDIENT garlic?"
    # ──────────────────────────────────────────────────────────────
    def predict_tail(
        self, head: str, relation: str, top_k: int = 10,
        filter_entities: set = None
    ) -> list[tuple[str, float]]:
        """
        Predict: (head, relation, ???) → ranked list of tail candidates.

        Args:
            head:     Entity name (e.g., "Chicken Biryani")
            relation: Relation name (e.g., "PROVIDES")
            top_k:    Number of results
            filter_entities: Optional set of entity names to restrict results to

        Returns:
            List of (entity_name, score) tuples, sorted by score ascending (best first)
        """
        if head not in self.entity_to_id:
            print(f"  WARNING: '{head}' not found in knowledge graph.")
            return []
        if relation not in self.relation_to_id:
            print(f"  WARNING: '{relation}' not found in relations.")
            return []

        scores = self._score_all_tails(head, relation)

        # Rank
        ranked_ids = np.argsort(scores)
        results = []
        for eid in ranked_ids:
            name = self.id_to_entity[eid]
            if name == head:
                continue  # skip self
            if filter_entities and name not in filter_entities:
                continue
            results.append((name, float(scores[eid])))
            if len(results) >= top_k:
                break

        return results

    def predict_head(
        self, relation: str, tail: str, top_k: int = 10,
        filter_entities: set = None
    ) -> list[tuple[str, float]]:
        """
        Predict: (???, relation, tail) → ranked list of head candidates.

        Example: (???, BENEFITS_FROM, iron) → [anemia, ...]
        """
        if tail not in self.entity_to_id:
            print(f"  WARNING: '{tail}' not found in knowledge graph.")
            return []
        if relation not in self.relation_to_id:
            print(f"  WARNING: '{relation}' not found in relations.")
            return []

        scores = self._score_all_heads(relation, tail)

        ranked_ids = np.argsort(scores)
        results = []
        for eid in ranked_ids:
            name = self.id_to_entity[eid]
            if name == tail:
                continue
            if filter_entities and name not in filter_entities:
                continue
            results.append((name, float(scores[eid])))
            if len(results) >= top_k:
                break

        return results

    # ──────────────────────────────────────────────────────────────
    #  QUERY 2: Entity Similarity
    #  "What recipes are most similar to Chicken Biryani?"
    # ──────────────────────────────────────────────────────────────
    def find_similar(
        self, entity: str, top_k: int = 10,
        filter_entities: set = None
    ) -> list[tuple[str, float]]:
        """
        Find entities closest in embedding space (L2 distance).

        This works because RotatE pushes entities that participate
        in similar relation patterns to similar regions of the space.
        """
        if entity not in self.entity_to_id:
            print(f"  WARNING: '{entity}' not found.")
            return []

        eid = self.entity_to_id[entity]
        query_emb = self.entity_embs[eid]

        # L2 distance to all entities
        diffs = self.entity_embs - query_emb
        distances = np.linalg.norm(diffs, axis=1)

        ranked_ids = np.argsort(distances)
        results = []
        for rid in ranked_ids:
            name = self.id_to_entity[rid]
            if name == entity:
                continue
            if filter_entities and name not in filter_entities:
                continue
            results.append((name, float(distances[rid])))
            if len(results) >= top_k:
                break

        return results

    # ──────────────────────────────────────────────────────────────
    #  QUERY 3: Multi-Hop Recommendation
    #  "I have anemia — what recipes should I eat?"
    #
    #  This is the killer feature of the knowledge graph.
    #  It chains: ailment → BENEFITS_FROM → nutrient ← PROVIDES ← recipe
    # ──────────────────────────────────────────────────────────────
    def recommend_for_ailment(
        self, ailment: str, top_k: int = 10,
        recipe_names: set = None
    ) -> list[tuple[str, float, list[str]]]:
        """
        Multi-hop inference:
          ailment → BENEFITS_FROM → nutrients → PROVIDES ← recipes

        Args:
            ailment:      e.g., "anemia"
            top_k:        Number of recipes to return
            recipe_names: Set of all recipe names (to filter results)

        Returns:
            List of (recipe_name, aggregate_score, [nutrients_provided]) tuples
        """
        if ailment not in self.entity_to_id:
            print(f"  WARNING: '{ailment}' not found.")
            return []

        # Step 1: What nutrients does this ailment benefit from?
        print(f"\n  Step 1: Finding nutrients that help with '{ailment}'...")
        nutrient_results = self.predict_tail(ailment, "BENEFITS_FROM", top_k=20)

        # Filter to actual nutrient entities (not recipes, etc.)
        # We use a heuristic: nutrients are typically short, lowercase names
        nutrients = []
        for name, score in nutrient_results:
            # Simple heuristic — in production you'd tag entity types
            if name[0].islower() and name not in self.relation_to_id:
                nutrients.append((name, score))
        nutrients = nutrients[:10]  # top 10 nutrients

        if not nutrients:
            print("  No nutrients found for this ailment.")
            return []

        print(f"  Found nutrients: {[n for n, _ in nutrients]}")

        # Step 2: For each nutrient, find recipes that PROVIDE it
        print(f"  Step 2: Finding recipes that provide these nutrients...")
        recipe_scores = {}   # recipe_name → aggregate score
        recipe_nutrients = {}  # recipe_name → list of nutrients it provides

        for nutrient_name, nutrient_score in nutrients:
            # (???, PROVIDES, nutrient) — which recipes provide this?
            recipe_results = self.predict_head("PROVIDES", nutrient_name, top_k=50)

            for recipe_name, recipe_score in recipe_results:
                if recipe_names and recipe_name not in recipe_names:
                    continue

                # Aggregate: lower total score = better
                # Weight by how relevant the nutrient is to the ailment
                combined = recipe_score + nutrient_score
                if recipe_name not in recipe_scores:
                    recipe_scores[recipe_name] = 0.0
                    recipe_nutrients[recipe_name] = []

                recipe_scores[recipe_name] += combined
                recipe_nutrients[recipe_name].append(nutrient_name)

        # Step 3: Rank recipes by aggregate score (more nutrients covered = better)
        # We normalize by number of nutrients to reward breadth
        final_scores = []
        for recipe_name, total_score in recipe_scores.items():
            num_nutrients = len(recipe_nutrients[recipe_name])
            # Lower is better, but more nutrients is better → divide by count
            normalized = total_score / (num_nutrients ** 1.5)  # reward coverage
            final_scores.append((
                recipe_name,
                normalized,
                recipe_nutrients[recipe_name]
            ))

        final_scores.sort(key=lambda x: x[1])
        return final_scores[:top_k]

    # ──────────────────────────────────────────────────────────────
    #  QUERY 4: Entity Exploration
    #  "Tell me everything about 'iron' in the graph"
    # ──────────────────────────────────────────────────────────────
    def explore_entity(self, entity: str, top_k: int = 5):
        """
        For a given entity, find its most likely connections
        across ALL relation types in both directions.
        """
        if entity not in self.entity_to_id:
            print(f"  WARNING: '{entity}' not found.")
            return

        print(f"\n{'='*60}")
        print(f"EXPLORING: {entity}")
        print(f"{'='*60}")

        # As tail: (entity, relation, ???) for each relation
        print(f"\n  Outgoing edges ('{entity}' as head):")
        for rel_name in self.relation_to_id:
            results = self.predict_tail(entity, rel_name, top_k=top_k)
            if results:
                top_str = ", ".join(f"{n} ({s:.2f})" for n, s in results[:3])
                print(f"    --[{rel_name}]--> {top_str}")

        # As head: (???, relation, entity) for each relation
        print(f"\n  Incoming edges ('{entity}' as tail):")
        for rel_name in self.relation_to_id:
            results = self.predict_head(rel_name, entity, top_k=top_k)
            if results:
                top_str = ", ".join(f"{n} ({s:.2f})" for n, s in results[:3])
                print(f"    <--[{rel_name}]-- {top_str}")

        # Nearest neighbors
        print(f"\n  Nearest entities in embedding space:")
        similar = self.find_similar(entity, top_k=top_k)
        for name, dist in similar:
            print(f"    {name:30s}  distance={dist:.4f}")

    # ──────────────────────────────────────────────────────────────
    #  QUERY 5: User Preference Recommendation
    #  "I liked Biryani and Pad Thai, what else would I like?"
    # ──────────────────────────────────────────────────────────────
    def recommend_from_likes(
        self, liked_recipes: list[str], top_k: int = 10,
        recipe_names: set = None
    ) -> list[tuple[str, float]]:
        """
        Average the embeddings of liked recipes, find nearest recipes.
        Simple but effective collaborative-filtering-style approach.
        """
        valid_ids = []
        for recipe in liked_recipes:
            if recipe in self.entity_to_id:
                valid_ids.append(self.entity_to_id[recipe])
            else:
                print(f"  WARNING: '{recipe}' not found, skipping.")

        if not valid_ids:
            return []

        # Average embedding of liked recipes
        liked_embs = self.entity_embs[valid_ids]
        centroid = liked_embs.mean(axis=0)

        # Distance to all entities
        diffs = self.entity_embs - centroid
        distances = np.linalg.norm(diffs, axis=1)

        ranked_ids = np.argsort(distances)
        results = []
        liked_set = set(liked_recipes)
        for rid in ranked_ids:
            name = self.id_to_entity[rid]
            if name in liked_set:
                continue
            if recipe_names and name not in recipe_names:
                continue
            results.append((name, float(distances[rid])))
            if len(results) >= top_k:
                break

        return results


# ══════════════════════════════════════════════════════════════════
#  PART 3: CLI INTERFACE
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RotatE Knowledge Graph: Train + Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python train_and_infer.py train --triples triples.tsv --epochs 300

  # Find similar recipes
  python train_and_infer.py similar --recipe "Chicken Biryani" --top 10

  # Recommend recipes for an ailment (multi-hop!)
  python train_and_infer.py recommend --ailment "anemia" --top 10

  # Link prediction: what does a recipe provide?
  python train_and_infer.py predict --head "Salmon Teriyaki" --relation "PROVIDES"

  # Explore everything connected to an entity
  python train_and_infer.py explore --entity "iron"

  # Recommend based on user taste
  python train_and_infer.py likes --recipes "Chicken Biryani" "Pad Thai" --top 10
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Train command ─────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train RotatE model")
    train_parser.add_argument("--triples", default="triples.tsv", help="Path to triples TSV")
    train_parser.add_argument("--output", default="trained_model", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    train_parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--num-negs", type=int, default=64, help="Negative samples per positive")

    # ── Similar command ───────────────────────────────────────────
    sim_parser = subparsers.add_parser("similar", help="Find similar recipes")
    sim_parser.add_argument("--recipe", required=True, help="Recipe name")
    sim_parser.add_argument("--top", type=int, default=10, help="Number of results")
    sim_parser.add_argument("--model-dir", default="trained_model")

    # ── Recommend command ─────────────────────────────────────────
    rec_parser = subparsers.add_parser("recommend", help="Recommend recipes for ailment")
    rec_parser.add_argument("--ailment", required=True, help="Ailment name")
    rec_parser.add_argument("--top", type=int, default=10, help="Number of results")
    rec_parser.add_argument("--model-dir", default="trained_model")

    # ── Predict command ───────────────────────────────────────────
    pred_parser = subparsers.add_parser("predict", help="Link prediction")
    pred_parser.add_argument("--head", help="Head entity")
    pred_parser.add_argument("--relation", required=True, help="Relation type")
    pred_parser.add_argument("--tail", help="Tail entity (if predicting head)")
    pred_parser.add_argument("--top", type=int, default=10, help="Number of results")
    pred_parser.add_argument("--model-dir", default="trained_model")

    # ── Explore command ───────────────────────────────────────────
    exp_parser = subparsers.add_parser("explore", help="Explore entity connections")
    exp_parser.add_argument("--entity", required=True, help="Entity to explore")
    exp_parser.add_argument("--model-dir", default="trained_model")

    # ── Likes command ─────────────────────────────────────────────
    likes_parser = subparsers.add_parser("likes", help="Recommend from liked recipes")
    likes_parser.add_argument("--recipes", nargs="+", required=True, help="Liked recipe names")
    likes_parser.add_argument("--top", type=int, default=10, help="Number of results")
    likes_parser.add_argument("--model-dir", default="trained_model")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # ── Execute commands ──────────────────────────────────────────
    if args.command == "train":
        train(
            triples_path=args.triples,
            output_dir=args.output,
            epochs=args.epochs,
            embedding_dim=args.dim,
            batch_size=args.batch_size,
            lr=args.lr,
            num_negs=args.num_negs,
        )
        return

    # All inference commands need the trained model
    kg = KnowledgeGraphInference(args.model_dir)

    # Build recipe name set for filtering (optional)
    recipe_json_path = "sample_recipe_data.json"
    recipe_names = None
    if os.path.exists(recipe_json_path):
        with open(recipe_json_path) as f:
            recipe_names = {r["name"] for r in json.load(f)["recipes"]}

    if args.command == "similar":
        print(f"\nRecipes similar to '{args.recipe}':")
        results = kg.find_similar(args.recipe, top_k=args.top, filter_entities=recipe_names)
        for i, (name, dist) in enumerate(results, 1):
            print(f"  {i:2d}. {name:35s}  distance={dist:.4f}")

    elif args.command == "recommend":
        print(f"\nRecommended recipes for '{args.ailment}':")
        results = kg.recommend_for_ailment(args.ailment, top_k=args.top, recipe_names=recipe_names)
        for i, (name, score, nutrients) in enumerate(results, 1):
            nutrient_str = ", ".join(nutrients)
            print(f"  {i:2d}. {name:35s}  score={score:.4f}  nutrients=[{nutrient_str}]")

    elif args.command == "predict":
        if args.head:
            print(f"\nPredicting: ({args.head}, {args.relation}, ???)")
            results = kg.predict_tail(args.head, args.relation, top_k=args.top)
            for i, (name, score) in enumerate(results, 1):
                print(f"  {i:2d}. {name:35s}  score={score:.4f}")
        elif args.tail:
            print(f"\nPredicting: (???, {args.relation}, {args.tail})")
            results = kg.predict_head(args.relation, args.tail, top_k=args.top)
            for i, (name, score) in enumerate(results, 1):
                print(f"  {i:2d}. {name:35s}  score={score:.4f}")
        else:
            print("ERROR: Specify either --head or --tail for prediction.")

    elif args.command == "explore":
        kg.explore_entity(args.entity)

    elif args.command == "likes":
        print(f"\nBased on your likes: {args.recipes}")
        results = kg.recommend_from_likes(args.recipes, top_k=args.top, recipe_names=recipe_names)
        for i, (name, dist) in enumerate(results, 1):
            print(f"  {i:2d}. {name:35s}  distance={dist:.4f}")


if __name__ == "__main__":
    main()
