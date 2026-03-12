"""
Module 2 - Knowledge Graph Embeddings
Trains TransE / RotatE / ComplEx via PyKEEN on the cancer KG.
Enables link prediction: f(h, r, t) = ||h + r - t||
Evaluates with Mean Rank and Hits@k metrics.
"""

import torch
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from pykeen.pipeline import pipeline
from pykeen.triples  import TriplesFactory
from pykeen.models   import TransE, RotatE, ComplEx


@dataclass
class EmbeddingResult:
    model_name:  str
    mean_rank:   float
    hits_at_1:   float
    hits_at_3:   float
    hits_at_10:  float
    model:       object = None


# ── Sample triples (matches graph_builder.py sample data) ────────────────────
SAMPLE_TRIPLES = [
    ("Gefitinib",  "TREATS",          "Lung Cancer"),
    ("Cisplatin",  "TREATS",          "Lung Cancer"),
    ("Docetaxel",  "TREATS",          "Lung Cancer"),
    ("Gefitinib",  "TARGETS",         "EGFR"),
    ("Cisplatin",  "TARGETS",         "ALK"),
    ("Cisplatin",  "TARGETS",         "KRAS"),
    ("TP53",       "ASSOCIATED_WITH", "Lung Cancer"),
    ("KRAS",       "ASSOCIATED_WITH", "Colorectal Cancer"),
    ("Gefitinib",  "INHIBITS",        "EGFR"),
    ("Imatinib",   "TREATS",          "Leukemia"),
    ("Imatinib",   "TARGETS",         "BCR-ABL"),
    ("BCR-ABL",    "CAUSES",          "Leukemia"),
]


def triples_to_factory(
    triples: List[Tuple[str, str, str]],
    train_ratio: float = 0.8,
) -> Tuple[TriplesFactory, TriplesFactory]:
    """Convert list of (h, r, t) strings into PyKEEN TriplesFactory."""
    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    tf = TriplesFactory.from_labeled_triples(df.values)

    n_train = int(len(triples) * train_ratio)
    training, testing = tf.split([n_train / len(triples),
                                   1 - n_train / len(triples)])
    print(f"[INFO] Triples — Train: {training.num_triples} | "
          f"Test: {testing.num_triples}")
    return training, testing


def train_transe(
    training: TriplesFactory,
    testing:  TriplesFactory,
    embedding_dim: int = 50,
    epochs: int = 100,
    output_dir: str = "models/kg_embeddings",
) -> EmbeddingResult:
    """
    Train TransE model.
    Scoring function: f(h,r,t) = -||h + r - t||
    Lower score = stronger/more valid relationship.
    """
    print("[INFO] Training TransE...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result = pipeline(
        training=training,
        testing=testing,
        model="TransE",
        model_kwargs=dict(embedding_dim=embedding_dim),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=0.01),
        loss="MarginRankingLoss",
        loss_kwargs=dict(margin=1.0),
        training_kwargs=dict(
            num_epochs=epochs,
            batch_size=32,
            use_tqdm_batch=False,
        ),
        evaluation_kwargs=dict(batch_size=32),
        random_seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    result.save_to_directory(output_dir)
    metrics = result.metric_results.to_flat_dict()

    emb_result = EmbeddingResult(
        model_name = "TransE",
        mean_rank  = metrics.get("both.realistic.mean_rank", 0),
        hits_at_1  = metrics.get("both.realistic.hits_at_1", 0),
        hits_at_3  = metrics.get("both.realistic.hits_at_3", 0),
        hits_at_10 = metrics.get("both.realistic.hits_at_10", 0),
        model      = result.model,
    )
    _print_metrics(emb_result)
    return emb_result


def train_rotate(
    training: TriplesFactory,
    testing:  TriplesFactory,
    embedding_dim: int = 50,
    epochs: int = 100,
    output_dir: str = "models/kg_embeddings_rotate",
) -> EmbeddingResult:
    """Train RotatE model — better for symmetric/antisymmetric relations."""
    print("[INFO] Training RotatE...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result = pipeline(
        training=training,
        testing=testing,
        model="RotatE",
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs, batch_size=32),
        random_seed=42,
    )
    result.save_to_directory(output_dir)
    metrics = result.metric_results.to_flat_dict()

    emb_result = EmbeddingResult(
        model_name = "RotatE",
        mean_rank  = metrics.get("both.realistic.mean_rank", 0),
        hits_at_1  = metrics.get("both.realistic.hits_at_1", 0),
        hits_at_3  = metrics.get("both.realistic.hits_at_3", 0),
        hits_at_10 = metrics.get("both.realistic.hits_at_10", 0),
        model      = result.model,
    )
    _print_metrics(emb_result)
    return emb_result


def predict_link(
    head: str,
    relation: str,
    tail: str,
    model,
    triples_factory: TriplesFactory,
) -> float:
    """
    Score a candidate triple using the trained embedding model.
    Lower score = more likely/valid relationship (TransE).
    """
    try:
        h_id = triples_factory.entity_to_id[head]
        r_id = triples_factory.relation_to_id[relation]
        t_id = triples_factory.entity_to_id[tail]
    except KeyError as e:
        print(f"[WARN] Entity/relation not in vocabulary: {e}")
        return float("inf")

    h_tensor = torch.tensor([h_id])
    r_tensor = torch.tensor([r_id])
    t_tensor = torch.tensor([t_id])

    model.eval()
    with torch.no_grad():
        score = model.score_hrt(
            torch.stack([h_tensor, r_tensor, t_tensor], dim=1)
        ).item()
    return score


def compare_models(triples: List[Tuple[str, str, str]]) -> None:
    """Train both TransE and RotatE and compare their metrics."""
    training, testing = triples_to_factory(triples)
    transe = train_transe(training, testing, epochs=50)
    rotate = train_rotate(training, testing, epochs=50)

    print("\n── Model Comparison ─────────────────────────────────")
    print(f"{'Model':<12} {'MeanRank':>10} {'H@1':>8} {'H@3':>8} {'H@10':>8}")
    print("-" * 50)
    for r in [transe, rotate]:
        print(f"{r.model_name:<12} {r.mean_rank:>10.2f} "
              f"{r.hits_at_1:>8.4f} {r.hits_at_3:>8.4f} "
              f"{r.hits_at_10:>8.4f}")


def _print_metrics(result: EmbeddingResult):
    print(f"\n── {result.model_name} Results ──────────────────────")
    print(f"  Mean Rank : {result.mean_rank:.2f}")
    print(f"  Hits@1    : {result.hits_at_1:.4f}")
    print(f"  Hits@3    : {result.hits_at_3:.4f}")
    print(f"  Hits@10   : {result.hits_at_10:.4f}")


if __name__ == "__main__":
    training, testing = triples_to_factory(SAMPLE_TRIPLES)
    result = train_transe(training, testing, epochs=100)

    # Test link prediction
    print("\n── Link Prediction Scores ───────────────────────────")
    test_pairs = [
        ("Cisplatin",  "TREATS", "Lung Cancer"),
        ("Gefitinib",  "TREATS", "Leukemia"),      # false — lower score expected
    ]
    for h, r, t in test_pairs:
        score = predict_link(h, r, t, result.model, training)
        print(f"  Score({h}, {r}, {t}): {score:.6f}")