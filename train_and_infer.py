"""
RotatE Knowledge Graph: Training + Inference Pipeline (v2)
==========================================================
Integrates with mined_config.json produced by mine_knowledge.py.
Supports loading config and triples from S3 URLs or HTTP(S) URLs.

Changes from v1:
  - Loads mined_config.json for entity metadata (descriptions, types, normalization)
  - Nutrient and ailment nodes have human-readable descriptions
  - Entity type awareness: knows which entities are recipes vs ingredients vs nutrients
  - Ingredient normalization: queries are normalized before lookup
  - Substitution-aware inference: can find substitutes via the graph
  - S3/URL support: --config and --triples accept URLs (auto-downloads + caches)
  - New commands: ailments, nutrients, substitutes, describe

Usage:
    # Train with local files
    python train_and_infer.py train --triples triples.tsv --epochs 300 --dim 256

    # Train with S3 URLs (auto-downloads)
    python train_and_infer.py train \\
        --triples https://your-bucket.s3.amazonaws.com/vitalbites/triples.tsv \\
        --epochs 300 --dim 256

    # Inference with S3 config
    python train_and_infer.py recommend --ailment anemia --top 10 \\
        --config https://your-bucket.s3.amazonaws.com/vitalbites/mined_config.json

    # Also supports s3:// protocol (requires boto3)
    python train_and_infer.py train \\
        --triples s3://your-bucket/vitalbites/triples.tsv \\
        --config s3://your-bucket/vitalbites/mined_config.json

    # Standard inference
    python train_and_infer.py recommend  --ailment anemia --top 10
    python train_and_infer.py similar    --recipe "Biryani" --top 10
    python train_and_infer.py ailments
    python train_and_infer.py nutrients
    python train_and_infer.py substitutes --ingredient butter
    python train_and_infer.py describe    --entity iron

Requirements:
    pip install pykeen torch pandas numpy
    pip install boto3   (only if using s3:// URLs)

Author: Brian / LeeroyChainkins AI — VitalBites Project
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════
#  S3 / URL DOWNLOAD HELPER
# ══════════════════════════════════════════════════════════════════

# Track files downloaded from URLs so we can clean them up after training
_downloaded_files = set()


def download_if_url(path_or_url: str, local_name: str = None) -> str:
    """
    If path_or_url is a URL (http/https/s3), download to local file and return local path.
    If it's already a local path, return as-is.
    Downloaded files are tracked for cleanup via cleanup_downloads().
    """
    if not path_or_url:
        return path_or_url

    # Already a local file
    if not path_or_url.startswith(("http://", "https://", "s3://")):
        return path_or_url

    # Determine local filename
    if local_name is None:
        local_name = path_or_url.rstrip("/").split("/")[-1].split("?")[0]
        if not local_name:
            local_name = "downloaded_file"

    # Skip download if already exists locally
    if os.path.exists(local_name):
        file_size = os.path.getsize(local_name)
        if file_size > 0:
            print(f"  Using cached {local_name} ({file_size:,} bytes)")
            _downloaded_files.add(local_name)
            return local_name

    # S3 native URL (s3://bucket/key)
    if path_or_url.startswith("s3://"):
        try:
            import boto3
        except ImportError:
            print("ERROR: boto3 required for s3:// URLs. Install with: pip install boto3")
            sys.exit(1)

        parts = path_or_url.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        print(f"  Downloading s3://{bucket}/{key} → {local_name}...")
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, local_name)
        print(f"  Downloaded {os.path.getsize(local_name):,} bytes")
        _downloaded_files.add(local_name)
        return local_name

    # HTTP(S) URL (including S3 pre-signed or public URLs)
    print(f"  Downloading {path_or_url}")
    print(f"    → {local_name}...")
    urllib.request.urlretrieve(path_or_url, local_name)
    print(f"  Downloaded {os.path.getsize(local_name):,} bytes")
    _downloaded_files.add(local_name)
    return local_name


def cleanup_downloads():
    """Remove all files that were downloaded from URLs during this session."""
    if not _downloaded_files:
        return
    print(f"\n  Cleaning up {len(_downloaded_files)} downloaded file(s)...")
    for fpath in list(_downloaded_files):
        try:
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                os.remove(fpath)
                print(f"    Removed {fpath} ({size:,} bytes)")
            _downloaded_files.discard(fpath)
        except OSError as e:
            print(f"    WARNING: Could not remove {fpath}: {e}")


def upload_to_s3(local_path: str, s3_uri: str):
    """Upload a local file or directory to S3."""
    import boto3

    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    s3 = boto3.client("s3")

    if os.path.isdir(local_path):
        file_count = 0
        for root, dirs, files in os.walk(local_path):
            for fname in files:
                local_file = os.path.join(root, fname)
                rel_path = os.path.relpath(local_file, local_path)
                s3_key = f"{prefix}/{rel_path}" if prefix else rel_path
                s3.upload_file(local_file, bucket, s3_key)
                file_count += 1
        print(f"  Uploaded {file_count} files to s3://{bucket}/{prefix}/")
    else:
        s3_key = f"{prefix}/{os.path.basename(local_path)}" if prefix else os.path.basename(local_path)
        s3.upload_file(local_path, bucket, s3_key)
        print(f"  Uploaded {local_path} to s3://{bucket}/{s3_key}")


def detect_sagemaker() -> dict | None:
    """
    Detect if running inside a SageMaker Training Job.
    Returns environment info dict or None if not in SageMaker.

    SageMaker sets these environment variables:
      SM_MODEL_DIR       — where to save model artifacts (copied to S3 after job)
      SM_CHANNEL_TRAINING — where input data is mounted
      SM_OUTPUT_DATA_DIR  — additional output artifacts
      SM_NUM_GPUS        — number of GPUs available
    """
    sm_model_dir = os.environ.get("SM_MODEL_DIR")
    if not sm_model_dir:
        return None

    env = {
        "model_dir": sm_model_dir,
        "training_dir": os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        "config_dir": os.environ.get("SM_CHANNEL_CONFIG", "/opt/ml/input/data/config"),
        "output_dir": os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
        "num_gpus": int(os.environ.get("SM_NUM_GPUS", "0")),
    }

    print("=" * 70)
    print("SAGEMAKER TRAINING ENVIRONMENT DETECTED")
    print("=" * 70)
    for k, v in env.items():
        print(f"  {k}: {v}")

    return env


# ══════════════════════════════════════════════════════════════════
#  ENTITY METADATA CATALOG
# ══════════════════════════════════════════════════════════════════

class EntityCatalog:
    """
    Loads mined_config.json and provides:
      - Entity type classification (recipe, ingredient, nutrient, ailment, etc.)
      - Human-readable descriptions for nutrients and ailments
      - Ingredient normalization (raw name → canonical)
      - Nutrient normalization (raw name → canonical)
      - Substitution lookup
      - Ailment → nutrient needs/avoid mapping
    """

    def __init__(self, config_path: str = "mined_config.json"):
        self.config = {}
        self.ingredient_normalize = {}
        self.nutrient_normalize = {}
        self.substitutions = {}
        self.ailment_map = {}
        self.nutrient_descriptions = {}
        self.ailment_descriptions = {}
        self.health_labels = {}
        self.micro_to_nutrient = {}

        # Download from URL if needed
        config_path = download_if_url(config_path, "mined_config.json")

        if os.path.exists(config_path):
            print(f"  Loading entity catalog from {config_path}...")
            with open(config_path) as f:
                self.config = json.load(f)

            self.ingredient_normalize = self.config.get("INGREDIENT_NORMALIZE", {})
            self.nutrient_normalize = self.config.get("NUTRIENT_NORMALIZE", {})
            self.substitutions = self.config.get("SUBSTITUTIONS", {})
            self.ailment_map = self.config.get("AILMENT_NUTRIENT_MAP", {})
            self.health_labels = self.config.get("HEALTH_DESCRIPTION_TO_LABEL", {})
            self.micro_to_nutrient = self.config.get("MICRO_DESCRIPTION_TO_NUTRIENT", {})

            self._build_nutrient_descriptions()
            self._build_ailment_descriptions()

            print(f"    Ingredients normalizable: {len(self.ingredient_normalize)}")
            print(f"    Nutrients normalizable:   {len(self.nutrient_normalize)}")
            print(f"    Substitutions loaded:     {len(self.substitutions)}")
            print(f"    Ailments loaded:          {len(self.ailment_map)}")
        else:
            print(f"  NOTE: {config_path} not found. Running without entity metadata.")
            print(f"         Descriptions, normalization, and type filtering unavailable.")

    def _build_nutrient_descriptions(self):
        """Build human-readable descriptions for each nutrient from the mined data."""
        nutrient_to_descs = {}
        for desc_text, nutrient_name in self.micro_to_nutrient.items():
            canonical = self.normalize_nutrient(nutrient_name)
            if canonical not in nutrient_to_descs:
                nutrient_to_descs[canonical] = []
            clean = desc_text.strip().rstrip(".")
            if clean.startswith("it "):
                clean = clean[3:]
            if clean.startswith("is "):
                clean = clean[3:]
            nutrient_to_descs[canonical].append(clean)
        self.nutrient_descriptions = nutrient_to_descs

    def _build_ailment_descriptions(self):
        """Build human-readable descriptions for each ailment."""
        for ailment, info in self.ailment_map.items():
            needs = info.get("needs", [])
            avoid = info.get("avoid", [])
            needs_canonical = [self.normalize_nutrient(n) for n in needs]
            avoid_canonical = [self.normalize_nutrient(n) for n in avoid]
            parts = []
            if needs_canonical:
                parts.append(f"Benefits from: {', '.join(needs_canonical)}")
            if avoid_canonical:
                parts.append(f"Should limit: {', '.join(avoid_canonical)}")
            self.ailment_descriptions[ailment] = "; ".join(parts)

    def normalize_ingredient(self, name: str) -> str:
        """Normalize an ingredient name to its canonical form."""
        name_lower = name.lower().strip()
        if name_lower in self.ingredient_normalize:
            return self.ingredient_normalize[name_lower]
        simplified = re.sub(r"\b(fresh|dried|ground|chopped|minced|frozen)\b", "", name_lower).strip()
        simplified = re.sub(r"\s+", " ", simplified)
        if simplified in self.ingredient_normalize:
            return self.ingredient_normalize[simplified]
        return name_lower.replace(" ", "_")

    def normalize_nutrient(self, name: str) -> str:
        """Normalize a nutrient name to its canonical form."""
        if name in self.nutrient_normalize:
            result = self.nutrient_normalize[name]
            return result if result else name
        return name

    def get_nutrient_description(self, nutrient: str) -> str:
        """Get a human-readable description for a nutrient."""
        canonical = self.normalize_nutrient(nutrient)
        descs = self.nutrient_descriptions.get(canonical, [])
        if not descs:
            descs = self.nutrient_descriptions.get(nutrient, [])
        return descs[0] if descs else ""

    def get_ailment_description(self, ailment: str) -> str:
        """Get a human-readable description for an ailment."""
        return self.ailment_descriptions.get(ailment, "")

    def get_ailment_nutrients(self, ailment: str) -> dict:
        """Get the needs/avoid nutrient lists for an ailment."""
        info = self.ailment_map.get(ailment, {})
        return {
            "needs": [self.normalize_nutrient(n) for n in info.get("needs", [])],
            "avoid": [self.normalize_nutrient(n) for n in info.get("avoid", [])],
        }

    def get_substitutes(self, ingredient: str) -> list[str]:
        """Get known substitutes for an ingredient."""
        canonical = self.normalize_ingredient(ingredient)
        if canonical in self.substitutions:
            return self.substitutions[canonical]
        if ingredient in self.substitutions:
            return self.substitutions[ingredient]
        return []

    def classify_entity(self, entity_name: str) -> str:
        """Classify an entity into a type."""
        if entity_name in self.ailment_map:
            return "ailment"
        all_nutrients = set(self.nutrient_normalize.keys()) | set(
            v for v in self.nutrient_normalize.values() if v
        )
        if entity_name in all_nutrients:
            return "nutrient"
        if entity_name in ("protein", "fiber", "fat", "saturated_fat", "sugar",
                           "sodium_high", "low-calorie", "moderate-calorie", "high-calorie"):
            return "macro_label"
        all_health_labels = set(self.health_labels.values()) if self.health_labels else set()
        if entity_name in all_health_labels:
            return "health_function"
        all_ingredients = set(self.ingredient_normalize.keys()) | set(self.ingredient_normalize.values())
        if entity_name in all_ingredients or entity_name.replace("_", " ") in all_ingredients:
            return "ingredient"
        if entity_name and entity_name[0].isupper():
            return "recipe"
        return "unknown"

    def describe_entity(self, entity_name: str) -> dict:
        """Get full metadata about any entity."""
        entity_type = self.classify_entity(entity_name)
        info = {"name": entity_name, "type": entity_type}

        if entity_type == "nutrient":
            desc = self.get_nutrient_description(entity_name)
            if desc:
                info["description"] = desc
            helped_ailments = []
            for ailment, adata in self.ailment_map.items():
                needs = [self.normalize_nutrient(n) for n in adata.get("needs", [])]
                if entity_name in needs or self.normalize_nutrient(entity_name) in needs:
                    helped_ailments.append(ailment)
            if helped_ailments:
                info["helps_with"] = helped_ailments

        elif entity_type == "ailment":
            desc = self.get_ailment_description(entity_name)
            if desc:
                info["description"] = desc
            nutrients = self.get_ailment_nutrients(entity_name)
            if nutrients["needs"]:
                info["needs_nutrients"] = nutrients["needs"]
            if nutrients["avoid"]:
                info["avoid_nutrients"] = nutrients["avoid"]

        elif entity_type == "ingredient":
            subs = self.get_substitutes(entity_name)
            if subs:
                info["substitutes"] = subs

        return info


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
    upload_s3: str = None,
    config_path: str = None,
):
    """
    Train RotatE on the extracted triples.

    SageMaker mode:
      When running inside a SageMaker Training Job, automatically:
        - Reads triples from SM_CHANNEL_TRAINING
        - Reads config from SM_CHANNEL_CONFIG
        - Saves model to SM_MODEL_DIR (auto-uploaded to S3 by SageMaker)

    Args:
        upload_s3: Optional S3 URI to upload trained model after training.
                   e.g., "s3://vitalbites/models/latest"
        config_path: Path to mined_config.json — copied into model dir for portability.
    """
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    print("=" * 70)
    print("TRAINING RotatE KNOWLEDGE GRAPH EMBEDDINGS")
    print("=" * 70)

    # ── SageMaker environment detection ───────────────────────────
    sm_env = detect_sagemaker()
    if sm_env:
        # Override paths with SageMaker directories
        triples_candidates = [
            os.path.join(sm_env["training_dir"], "triples.tsv"),
            os.path.join(sm_env["training_dir"], os.path.basename(triples_path)),
        ]
        for candidate in triples_candidates:
            if os.path.exists(candidate):
                triples_path = candidate
                break

        # Config from SageMaker channel
        config_candidates = [
            os.path.join(sm_env["config_dir"], "mined_config.json"),
            os.path.join(sm_env["training_dir"], "mined_config.json"),
        ]
        for candidate in config_candidates:
            if os.path.exists(candidate):
                config_path = candidate
                break

        # Save model to SageMaker model dir
        output_dir = sm_env["model_dir"]

    # ── Download from URL if needed ───────────────────────────────
    triples_path = download_if_url(triples_path, "triples.tsv")

    print(f"\nLoading triples from {triples_path}...")
    tf = TriplesFactory.from_path(triples_path)
    print(f"  Total triples:    {tf.num_triples}")
    print(f"  Unique entities:  {tf.num_entities}")
    print(f"  Unique relations: {tf.num_relations}")

    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=seed)
    print(f"  Training split:   {training.num_triples}")
    print(f"  Validation split: {validation.num_triples}")
    print(f"  Test split:       {testing.num_triples}")

    print(f"\nTraining RotatE (dim={embedding_dim}, epochs={epochs})...")

    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model="RotatE",
        model_kwargs={"embedding_dim": embedding_dim},
        training_loop="sLCWA",
        training_kwargs={
            "num_epochs": epochs,
            "batch_size": batch_size,
            "checkpoint_name": "checkpoint.pt",
            "checkpoint_frequency": 50,
            "checkpoint_directory": output_dir,
        },
        negative_sampler="basic",
        negative_sampler_kwargs={"num_negs_per_pos": num_negs},
        optimizer="Adam",
        optimizer_kwargs={"lr": lr},
        evaluator_kwargs={"filtered": True},
        random_seed=seed,
    )

    # Evaluation metrics
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    metrics = result.metric_results
    mrr = metrics.get_metric("both.realistic.inverse_harmonic_mean_rank")
    h1 = metrics.get_metric("both.realistic.hits_at_1")
    h3 = metrics.get_metric("both.realistic.hits_at_3")
    h10 = metrics.get_metric("both.realistic.hits_at_10")

    print(f"  MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"  Hits@1:  {h1:.4f}    Hits@3:  {h3:.4f}    Hits@10: {h10:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    result.save_to_directory(output_dir)

    entity_to_id = training.entity_to_id
    relation_to_id = training.relation_to_id

    with open(os.path.join(output_dir, "entity_to_id.json"), "w") as f:
        json.dump(entity_to_id, f, indent=2)
    with open(os.path.join(output_dir, "relation_to_id.json"), "w") as f:
        json.dump(relation_to_id, f, indent=2)

    model = result.model
    entity_embs = model.entity_representations[0](
        indices=torch.arange(training.num_entities)
    ).detach().cpu().numpy()
    relation_embs = model.relation_representations[0](
        indices=torch.arange(training.num_relations)
    ).detach().cpu().numpy()

    np.save(os.path.join(output_dir, "entity_embeddings.npy"), entity_embs)
    np.save(os.path.join(output_dir, "relation_embeddings.npy"), relation_embs)

    # Copy mined_config.json into model dir for portability
    if config_path and os.path.exists(config_path):
        import shutil
        dest = os.path.join(output_dir, "mined_config.json")
        if os.path.abspath(config_path) != os.path.abspath(dest):
            shutil.copy2(config_path, dest)
            print(f"  Copied {config_path} → {dest}")

    # Save training metrics as JSON for programmatic access
    metrics_dict = {
        "mrr": float(mrr), "hits_at_1": float(h1),
        "hits_at_3": float(h3), "hits_at_10": float(h10),
        "epochs": epochs, "embedding_dim": embedding_dim,
        "num_triples": tf.num_triples, "num_entities": tf.num_entities,
        "num_relations": tf.num_relations,
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\nSaved to ./{output_dir}/")
    print(f"  Embeddings: {entity_embs.shape} entities, {relation_embs.shape} relations")

    # ── Upload to S3 if requested ─────────────────────────────────
    if upload_s3:
        print(f"\nUploading model to {upload_s3}...")
        upload_to_s3(output_dir, upload_s3)

    return result


# ══════════════════════════════════════════════════════════════════
#  PART 2: INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════

class KnowledgeGraphInference:
    """
    Inference engine with entity catalog integration.
    Provides type-aware filtering, normalization, and descriptions.
    """

    def __init__(self, model_dir: str = "trained_model", config_path: str = "mined_config.json"):
        print(f"Loading trained model from {model_dir}...")

        with open(os.path.join(model_dir, "entity_to_id.json")) as f:
            self.entity_to_id = json.load(f)
        with open(os.path.join(model_dir, "relation_to_id.json")) as f:
            self.relation_to_id = json.load(f)

        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}

        self.entity_embs = np.load(os.path.join(model_dir, "entity_embeddings.npy"))
        self.relation_embs = np.load(os.path.join(model_dir, "relation_embeddings.npy"))

        dim = self.entity_embs.shape[1] // 2
        self.entity_complex = self.entity_embs[:, :dim] + 1j * self.entity_embs[:, dim:]
        self.relation_complex = self.relation_embs[:, :dim] + 1j * self.relation_embs[:, dim:]

        print(f"  {len(self.entity_to_id)} entities, {len(self.relation_to_id)} relations, dim={dim}")

        # Load catalog
        self.catalog = EntityCatalog(config_path)
        self._build_type_index()

    def _build_type_index(self):
        """Pre-classify all entities for fast type filtering."""
        self.entities_by_type = {}
        for entity_name in self.entity_to_id:
            etype = self.catalog.classify_entity(entity_name)
            if etype not in self.entities_by_type:
                self.entities_by_type[etype] = set()
            self.entities_by_type[etype].add(entity_name)

        print(f"  Entity type breakdown:")
        for etype, entities in sorted(self.entities_by_type.items(), key=lambda x: -len(x[1])):
            if entities:
                print(f"    {etype:20s} {len(entities):>8,}")

    def resolve_entity(self, name: str) -> str | None:
        """
        Resolve user input to a graph entity.
        Tries: exact → normalized ingredient → normalized nutrient → case variants → fuzzy.
        """
        if name in self.entity_to_id:
            return name

        # Normalized forms
        for variant in [
            self.catalog.normalize_ingredient(name),
            self.catalog.normalize_nutrient(name),
            name.replace(" ", "_"),
            name.replace("_", " "),
            name.lower(),
            name.lower().replace(" ", "_"),
            name.title(),
        ]:
            if variant and variant in self.entity_to_id:
                return variant

        # Fuzzy substring match
        name_lower = name.lower()
        matches = [e for e in self.entity_to_id if name_lower in e.lower()]
        if len(matches) == 1:
            return matches[0]
        if 1 < len(matches) <= 5:
            print(f"  Ambiguous '{name}'. Did you mean:")
            for m in matches:
                print(f"    - {m}")
            return None

        print(f"  WARNING: '{name}' not found in knowledge graph.")
        return None

    # ── Core scoring ──────────────────────────────────────────────

    def _score_all_tails(self, head_name: str, relation_name: str) -> np.ndarray:
        h = self.entity_complex[self.entity_to_id[head_name]]
        r = self.relation_complex[self.relation_to_id[relation_name]]
        diffs = self.entity_complex - (h * r)
        return np.sum(np.abs(diffs), axis=1)

    def _score_all_heads(self, relation_name: str, tail_name: str) -> np.ndarray:
        t = self.entity_complex[self.entity_to_id[tail_name]]
        r = self.relation_complex[self.relation_to_id[relation_name]]
        diffs = self.entity_complex - (t * np.conj(r))
        return np.sum(np.abs(diffs), axis=1)

    # ── Link prediction ───────────────────────────────────────────

    def predict_tail(self, head: str, relation: str, top_k: int = 10,
                     filter_types: set = None) -> list[tuple[str, float]]:
        head_r = self.resolve_entity(head)
        if not head_r or relation not in self.relation_to_id:
            return []
        scores = self._score_all_tails(head_r, relation)
        return self._rank(scores, exclude={head_r}, filter_types=filter_types, top_k=top_k)

    def predict_head(self, relation: str, tail: str, top_k: int = 10,
                     filter_types: set = None) -> list[tuple[str, float]]:
        tail_r = self.resolve_entity(tail)
        if not tail_r or relation not in self.relation_to_id:
            return []
        scores = self._score_all_heads(relation, tail_r)
        return self._rank(scores, exclude={tail_r}, filter_types=filter_types, top_k=top_k)

    def _rank(self, scores: np.ndarray, exclude: set = None,
              filter_types: set = None, top_k: int = 10) -> list[tuple[str, float]]:
        ranked_ids = np.argsort(scores)
        results = []
        for eid in ranked_ids:
            name = self.id_to_entity[eid]
            if exclude and name in exclude:
                continue
            if filter_types:
                if self.catalog.classify_entity(name) not in filter_types:
                    continue
            results.append((name, float(scores[eid])))
            if len(results) >= top_k:
                break
        return results

    # ── Entity similarity ─────────────────────────────────────────

    def find_similar(self, entity: str, top_k: int = 10,
                     filter_types: set = None) -> list[tuple[str, float]]:
        entity_r = self.resolve_entity(entity)
        if not entity_r:
            return []
        diffs = self.entity_embs - self.entity_embs[self.entity_to_id[entity_r]]
        distances = np.linalg.norm(diffs, axis=1)
        return self._rank(distances, exclude={entity_r}, filter_types=filter_types, top_k=top_k)

    # ── Multi-hop ailment recommendation ──────────────────────────

    def recommend_for_ailment(self, ailment: str, top_k: int = 10
                              ) -> list[tuple[str, float, list[str]]]:
        """ailment → BENEFITS_FROM → nutrients → PROVIDES ← recipes"""
        ailment_r = self.resolve_entity(ailment)
        if not ailment_r:
            return []

        known = self.catalog.get_ailment_nutrients(ailment_r)
        known_needs = known.get("needs", [])
        known_avoid = known.get("avoid", [])

        print(f"\n  Ailment: {ailment_r}")
        desc = self.catalog.get_ailment_description(ailment_r)
        if desc:
            print(f"  {desc}")

        # Resolve known nutrients
        nutrients = []
        for nutrient in known_needs:
            n_resolved = self.resolve_entity(nutrient)
            if n_resolved:
                nutrients.append((n_resolved, 0.0))
                ndesc = self.catalog.get_nutrient_description(nutrient)
                print(f"    Needs: {nutrient}" + (f" — {ndesc}" if ndesc else ""))

        if known_avoid:
            for nutrient in known_avoid:
                print(f"    Avoid: {nutrient}")

        # Supplement with predicted nutrients
        if len(nutrients) < 5 and "BENEFITS_FROM" in self.relation_to_id:
            predicted = self.predict_tail(ailment_r, "BENEFITS_FROM", top_k=20,
                                          filter_types={"nutrient", "macro_label"})
            existing = {n for n, _ in nutrients}
            for name, score in predicted:
                if name not in existing and name not in known_avoid:
                    nutrients.append((name, score))
                    if len(nutrients) >= 10:
                        break

        if not nutrients:
            print("  No nutrients found for this ailment.")
            return []

        print(f"\n  Searching for recipes providing: {[n for n, _ in nutrients]}")

        # Find recipes providing these nutrients
        recipe_scores = {}
        recipe_nutrients = {}

        for nutrient_name, nutrient_score in nutrients:
            results = self.predict_head("PROVIDES", nutrient_name, top_k=100,
                                        filter_types={"recipe"})
            for recipe_name, recipe_score in results:
                combined = recipe_score + nutrient_score
                if recipe_name not in recipe_scores:
                    recipe_scores[recipe_name] = 0.0
                    recipe_nutrients[recipe_name] = []
                recipe_scores[recipe_name] += combined
                recipe_nutrients[recipe_name].append(nutrient_name)

        final = []
        for recipe_name, total_score in recipe_scores.items():
            n_count = len(recipe_nutrients[recipe_name])
            normalized = total_score / (n_count ** 1.5)
            final.append((recipe_name, normalized, recipe_nutrients[recipe_name]))

        final.sort(key=lambda x: x[1])
        return final[:top_k]

    # ── Entity exploration (with descriptions) ────────────────────

    def explore_entity(self, entity: str, top_k: int = 5):
        entity_r = self.resolve_entity(entity)
        if not entity_r:
            return

        info = self.catalog.describe_entity(entity_r)

        print(f"\n{'='*60}")
        print(f"EXPLORING: {entity_r}")
        print(f"  Type: {info['type']}")
        if "description" in info:
            print(f"  Description: {info['description']}")
        if "needs_nutrients" in info:
            print(f"  Needs: {', '.join(info['needs_nutrients'])}")
        if "avoid_nutrients" in info:
            print(f"  Avoid: {', '.join(info['avoid_nutrients'])}")
        if "helps_with" in info:
            print(f"  Helps with: {', '.join(info['helps_with'])}")
        if "substitutes" in info:
            print(f"  Substitutes: {', '.join(info['substitutes'])}")
        print(f"{'='*60}")

        print(f"\n  Outgoing edges:")
        for rel_name in self.relation_to_id:
            results = self.predict_tail(entity_r, rel_name, top_k=top_k)
            if results:
                items = []
                for n, s in results[:3]:
                    ndesc = self.catalog.get_nutrient_description(n)
                    if ndesc and len(ndesc) < 60:
                        items.append(f"{n} ({s:.2f}) [{ndesc}]")
                    else:
                        items.append(f"{n} ({s:.2f})")
                print(f"    --[{rel_name}]--> {', '.join(items)}")

        print(f"\n  Incoming edges:")
        for rel_name in self.relation_to_id:
            results = self.predict_head(rel_name, entity_r, top_k=top_k)
            if results:
                top_str = ", ".join(f"{n} ({s:.2f})" for n, s in results[:3])
                print(f"    <--[{rel_name}]-- {top_str}")

        print(f"\n  Nearest entities:")
        similar = self.find_similar(entity_r, top_k=top_k)
        for name, dist in similar:
            etype = self.catalog.classify_entity(name)
            print(f"    {name:30s}  dist={dist:.4f}  type={etype}")

    # ── User preference recommendation ────────────────────────────

    def recommend_from_likes(self, liked_recipes: list[str], top_k: int = 10
                             ) -> list[tuple[str, float]]:
        valid_ids = []
        for recipe in liked_recipes:
            r = self.resolve_entity(recipe)
            if r:
                valid_ids.append(self.entity_to_id[r])
        if not valid_ids:
            return []

        centroid = self.entity_embs[valid_ids].mean(axis=0)
        distances = np.linalg.norm(self.entity_embs - centroid, axis=1)

        liked_resolved = {self.resolve_entity(r) for r in liked_recipes}
        return self._rank(distances, exclude=liked_resolved,
                         filter_types={"recipe"}, top_k=top_k)

    # ── Substitution lookup ───────────────────────────────────────

    def find_substitutes(self, ingredient: str, top_k: int = 10) -> list[dict]:
        """Find substitutes from catalog + graph + embedding similarity."""
        ingredient_r = self.resolve_entity(ingredient)
        results = []

        # Source 1: Catalog (validated)
        catalog_subs = self.catalog.get_substitutes(ingredient or "")
        for sub in catalog_subs:
            sub_r = self.resolve_entity(sub)
            results.append({"name": sub, "source": "validated", "in_graph": sub_r is not None})

        # Source 2: KG relation
        if ingredient_r and "SUBSTITUTES_FOR" in self.relation_to_id:
            predicted = self.predict_tail(ingredient_r, "SUBSTITUTES_FOR", top_k=top_k,
                                          filter_types={"ingredient"})
            existing = {r["name"] for r in results}
            for name, score in predicted:
                if name not in existing:
                    results.append({"name": name, "source": "predicted", "score": round(score, 4)})

        # Source 3: Embedding neighbors
        if ingredient_r:
            similar = self.find_similar(ingredient_r, top_k=top_k,
                                        filter_types={"ingredient"})
            existing = {r["name"] for r in results}
            for name, dist in similar:
                if name not in existing:
                    results.append({"name": name, "source": "embedding_similarity",
                                   "distance": round(dist, 4)})
                if len(results) >= top_k:
                    break

        return results[:top_k]


# ══════════════════════════════════════════════════════════════════
#  PART 3: CLI INTERFACE
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RotatE Knowledge Graph: Train + Inference (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_and_infer.py train --triples triples.tsv --epochs 300
  python train_and_infer.py recommend --ailment anemia --top 10
  python train_and_infer.py similar --recipe "Biryani" --top 10
  python train_and_infer.py predict --head "Biryani" --relation PROVIDES
  python train_and_infer.py explore --entity iron
  python train_and_infer.py likes --recipes "Biryani" "Pad Thai" --top 10
  python train_and_infer.py ailments
  python train_and_infer.py nutrients
  python train_and_infer.py substitutes --ingredient butter
  python train_and_infer.py describe --entity iron
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build all subparsers with shared args
    commands = {
        "train": "Train RotatE model",
        "similar": "Find similar recipes",
        "recommend": "Recommend recipes for ailment",
        "predict": "Link prediction",
        "explore": "Explore entity connections",
        "likes": "Recommend from liked recipes",
        "ailments": "List all ailments with descriptions",
        "nutrients": "List all nutrients with descriptions",
        "substitutes": "Find substitutes for an ingredient",
        "describe": "Describe any entity",
    }

    sps = {}
    for name, help_text in commands.items():
        sp = subparsers.add_parser(name, help=help_text)
        sp.add_argument("--config", default="mined_config.json", help="Path to mined_config.json")
        sp.add_argument("--model-dir", default="trained_model", help="Trained model directory")
        sps[name] = sp

    # Command-specific args
    sps["train"].add_argument("--triples", default="triples.tsv")
    sps["train"].add_argument("--output", default="trained_model")
    sps["train"].add_argument("--epochs", type=int, default=300)
    sps["train"].add_argument("--dim", type=int, default=256)
    sps["train"].add_argument("--batch-size", type=int, default=256)
    sps["train"].add_argument("--lr", type=float, default=1e-3)
    sps["train"].add_argument("--num-negs", type=int, default=64)
    sps["train"].add_argument("--upload-s3", default=None,
                               help="S3 URI to upload model after training (e.g., s3://vitalbites/models/latest)")

    sps["similar"].add_argument("--recipe", required=True)
    sps["similar"].add_argument("--top", type=int, default=10)

    sps["recommend"].add_argument("--ailment", required=True)
    sps["recommend"].add_argument("--top", type=int, default=10)

    sps["predict"].add_argument("--head")
    sps["predict"].add_argument("--relation", required=True)
    sps["predict"].add_argument("--tail")
    sps["predict"].add_argument("--top", type=int, default=10)

    sps["explore"].add_argument("--entity", required=True)

    sps["likes"].add_argument("--recipes", nargs="+", required=True)
    sps["likes"].add_argument("--top", type=int, default=10)

    sps["substitutes"].add_argument("--ingredient", required=True)
    sps["substitutes"].add_argument("--top", type=int, default=10)

    sps["describe"].add_argument("--entity", required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Resolve URLs for config and triples
    config_path = download_if_url(args.config, "mined_config.json")

    # ── Train (no catalog needed) ─────────────────────────────────
    if args.command == "train":
        try:
            train(
                triples_path=args.triples,
                output_dir=args.output,
                epochs=args.epochs,
                embedding_dim=args.dim,
                batch_size=args.batch_size,
                lr=args.lr,
                num_negs=args.num_negs,
                upload_s3=args.upload_s3,
                config_path=config_path,
            )
        finally:
            cleanup_downloads()
        return

    # ── Catalog-only commands (no trained model needed) ───────────
    if args.command == "ailments":
        catalog = EntityCatalog(config_path)
        print(f"\n{'='*70}")
        print(f"ALL AILMENTS ({len(catalog.ailment_map)} conditions)")
        print(f"{'='*70}")
        for ailment in sorted(catalog.ailment_map):
            info = catalog.ailment_map[ailment]
            needs = [catalog.normalize_nutrient(n) for n in info.get("needs", [])]
            avoid = [catalog.normalize_nutrient(n) for n in info.get("avoid", [])]
            print(f"\n  {ailment}")
            if needs:
                print(f"    Needs:  {', '.join(needs)}")
            if avoid:
                print(f"    Avoid:  {', '.join(avoid)}")
        return

    if args.command == "nutrients":
        catalog = EntityCatalog(config_path)
        print(f"\n{'='*70}")
        print(f"ALL NUTRIENTS ({len(catalog.nutrient_descriptions)} with descriptions)")
        print(f"{'='*70}")
        for nutrient in sorted(catalog.nutrient_descriptions):
            descs = catalog.nutrient_descriptions[nutrient]
            helped = []
            for ailment, adata in catalog.ailment_map.items():
                ailment_needs = [catalog.normalize_nutrient(n) for n in adata.get("needs", [])]
                if nutrient in ailment_needs:
                    helped.append(ailment)
            print(f"\n  {nutrient}")
            if descs:
                print(f"    Function: {descs[0]}")
                for extra in descs[1:]:
                    print(f"              {extra}")
            if helped:
                print(f"    Helps with: {', '.join(helped)}")
        return

    if args.command == "describe":
        catalog = EntityCatalog(config_path)
        info = catalog.describe_entity(args.entity)
        print(f"\n{'='*60}")
        for key, value in info.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(str(v) for v in value)}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*60}")
        return

    # ── Commands that need the trained model ──────────────────────
    kg = KnowledgeGraphInference(args.model_dir, config_path)

    if args.command == "similar":
        print(f"\nRecipes similar to '{args.recipe}':")
        results = kg.find_similar(args.recipe, top_k=args.top, filter_types={"recipe"})
        for i, (name, dist) in enumerate(results, 1):
            print(f"  {i:2d}. {name:40s}  distance={dist:.4f}")

    elif args.command == "recommend":
        print(f"\nRecommended recipes for '{args.ailment}':")
        results = kg.recommend_for_ailment(args.ailment, top_k=args.top)
        for i, (name, score, nutrients) in enumerate(results, 1):
            print(f"  {i:2d}. {name:40s}  score={score:.4f}")
            print(f"      nutrients: [{', '.join(nutrients)}]")

    elif args.command == "predict":
        if args.head:
            print(f"\nPredicting: ({args.head}, {args.relation}, ???)")
            results = kg.predict_tail(args.head, args.relation, top_k=args.top)
            for i, (name, score) in enumerate(results, 1):
                desc = kg.catalog.get_nutrient_description(name)
                extra = f"  — {desc}" if desc else ""
                print(f"  {i:2d}. {name:35s}  score={score:.4f}{extra}")
        elif args.tail:
            print(f"\nPredicting: (???, {args.relation}, {args.tail})")
            results = kg.predict_head(args.relation, args.tail, top_k=args.top)
            for i, (name, score) in enumerate(results, 1):
                print(f"  {i:2d}. {name:35s}  score={score:.4f}")
        else:
            print("ERROR: Specify either --head or --tail.")

    elif args.command == "explore":
        kg.explore_entity(args.entity)

    elif args.command == "likes":
        print(f"\nBased on your likes: {args.recipes}")
        results = kg.recommend_from_likes(args.recipes, top_k=args.top)
        for i, (name, dist) in enumerate(results, 1):
            print(f"  {i:2d}. {name:40s}  distance={dist:.4f}")

    elif args.command == "substitutes":
        print(f"\nSubstitutes for '{args.ingredient}':")
        results = kg.find_substitutes(args.ingredient, top_k=args.top)
        for i, sub in enumerate(results, 1):
            name = sub["name"]
            source = sub["source"]
            extras = []
            if "score" in sub:
                extras.append(f"score={sub['score']}")
            if "distance" in sub:
                extras.append(f"dist={sub['distance']}")
            if "in_graph" in sub:
                extras.append(f"in_graph={sub['in_graph']}")
            extra_str = f"  ({', '.join(extras)})" if extras else ""
            print(f"  {i:2d}. {name:30s}  [{source}]{extra_str}")


if __name__ == "__main__":
    main()

