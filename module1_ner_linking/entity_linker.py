"""
Module 1 - Entity Linker
Two-stage entity linking:
  Stage 1: Sentence-BERT + FAISS → top-k candidates
  Stage 2: Cross-encoder re-ranking (cosine + Levenshtein + popularity)
Links entities to NCBI Gene and DrugBank knowledge bases.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from Levenshtein import ratio as levenshtein_ratio
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class Entity:
    """A biomedical entity mention extracted from text."""
    text:        str
    label:       str          # Chemical | Disease | Gene
    start:       int
    end:         int
    linked_id:   Optional[str]  = None
    linked_name: Optional[str]  = None
    score:       float          = 0.0


@dataclass
class KnowledgeBase:
    """In-memory knowledge base of canonical entities."""
    ids:        List[str]         = field(default_factory=list)
    names:      List[str]         = field(default_factory=list)
    popularity: List[float]       = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None


# ── Scoring weights ───────────────────────────────────────────────────────────
W_COSINE      = 0.5
W_STRING      = 0.3
W_POPULARITY  = 0.2
TOP_K         = 10

BI_ENCODER_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EntityLinker:
    """
    Links surface-form entity mentions to canonical KB entries.

    Pipeline:
      1. Encode all KB names with Sentence-BERT → FAISS index
      2. For each mention, retrieve top-k candidates via ANN search
      3. Re-rank with cross-encoder + string similarity + popularity
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb           = kb
        self.bi_encoder   = SentenceTransformer(BI_ENCODER_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self.index        = None
        self._build_index()

    def _build_index(self):
        """Encode KB names and build FAISS flat-L2 index."""
        print("[INFO] Building FAISS index for entity linking...")
        if self.kb.embeddings is None:
            self.kb.embeddings = self.bi_encoder.encode(
                self.kb.names,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        dim         = self.kb.embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dim)   # Inner product = cosine (normalized)
        self.index.add(self.kb.embeddings)
        print(f"[INFO] FAISS index built: {self.index.ntotal} entities")

    def _candidate_generation(
        self, mention: str, top_k: int = TOP_K
    ) -> List[Tuple[int, float]]:
        """Stage 1: ANN search → top-k (index, cosine_score) pairs."""
        vec = self.bi_encoder.encode(
            [mention], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, indices = self.index.search(vec, top_k)
        return list(zip(indices[0].tolist(), scores[0].tolist()))

    def _rerank(
        self,
        mention: str,
        candidates: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """
        Stage 2: Re-rank candidates using:
          - Cross-encoder context similarity
          - Levenshtein string similarity
          - Entity popularity
        """
        pairs       = [[mention, self.kb.names[idx]] for idx, _ in candidates]
        ce_scores   = self.cross_encoder.predict(pairs)

        ranked = []
        for (idx, cosine), ce_score in zip(candidates, ce_scores):
            name       = self.kb.names[idx]
            str_sim    = levenshtein_ratio(mention.lower(), name.lower())
            popularity = self.kb.popularity[idx] if self.kb.popularity else 0.0

            final_score = (
                W_COSINE     * float(cosine)    +
                W_STRING     * str_sim          +
                W_POPULARITY * popularity
            )
            ranked.append((idx, final_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def link(self, mention: str) -> Tuple[str, str, float]:
        """
        Link a mention to the best KB entry.
        Returns (entity_id, entity_name, score).
        """
        candidates = self._candidate_generation(mention)
        ranked     = self._rerank(mention, candidates)
        best_idx, best_score = ranked[0]
        return (
            self.kb.ids[best_idx],
            self.kb.names[best_idx],
            best_score,
        )

    def link_batch(self, entities: List[Entity]) -> List[Entity]:
        """Link a list of Entity objects in place."""
        for ent in entities:
            ent.linked_id, ent.linked_name, ent.score = self.link(ent.text)
        return entities


def build_sample_kb() -> KnowledgeBase:
    """
    Sample KB with a few drug + disease entries for testing.
    Replace with full DrugBank / NCBI Gene data in production.
    """
    return KnowledgeBase(
        ids=[
            "DB00619", "DB01048", "DB00877",   # DrugBank
            "MESH:D008175", "MESH:D001943",    # MeSH diseases
            "NCBIGene:7157", "NCBIGene:4609",  # NCBI Gene
        ],
        names=[
            "Imatinib", "Abacavir", "Sirolimus",
            "Lung Cancer", "Breast Cancer",
            "TP53", "MYC",
        ],
        popularity=[0.9, 0.7, 0.6, 0.95, 0.92, 0.88, 0.80],
    )


if __name__ == "__main__":
    kb     = build_sample_kb()
    linker = EntityLinker(kb)

    test_mentions = ["lung cancer", "imatinib", "TP53 gene", "breast carcinoma"]
    for mention in test_mentions:
        eid, ename, score = linker.link(mention)
        print(f"  '{mention}' → [{eid}] {ename}  (score: {score:.3f})")