"""
Module 1 - End-to-End Pipeline
Connects NER → Entity Linking → Structured Output
Input:  raw biomedical text / PDF
Output: structured list of linked entities (Gene, Drug, Disease + IDs)
"""

import json
from pathlib import Path
from typing import List, Union
import fitz   # PyMuPDF — PDF parsing

from transformers import AutoTokenizer, AutoModelForTokenClassification
from .ner_model   import predict as ner_predict, ID2LABEL, MODEL_CHECKPOINT
from .entity_linker import Entity, EntityLinker, KnowledgeBase, build_sample_kb


ENTITY_LABELS = {"B-Chemical", "B-Disease", "I-Chemical", "I-Disease"}


def load_ner_model(model_path: str = MODEL_CHECKPOINT):
    """Load tokenizer + NER model from path or HuggingFace Hub."""
    print(f"[INFO] Loading NER model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForTokenClassification.from_pretrained(model_path)
    return tokenizer, model


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc  = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_entities_from_tokens(
    token_label_pairs: List[tuple],
) -> List[Entity]:
    """
    Convert (token, label) pairs from NER into Entity objects.
    Merges consecutive B-/I- tokens into single spans.
    """
    entities = []
    current_tokens, current_label = [], None
    pos = 0

    for token, label in token_label_pairs:
        if label.startswith("B-"):
            if current_tokens:
                entities.append(Entity(
                    text=" ".join(current_tokens),
                    label=current_label,
                    start=pos - len(" ".join(current_tokens)),
                    end=pos,
                ))
            current_tokens = [token]
            current_label  = label[2:]   # strip B-
        elif label.startswith("I-") and current_tokens:
            current_tokens.append(token)
        else:
            if current_tokens:
                entities.append(Entity(
                    text=" ".join(current_tokens),
                    label=current_label,
                    start=pos - len(" ".join(current_tokens)),
                    end=pos,
                ))
            current_tokens, current_label = [], None
        pos += len(token) + 1

    # flush last entity
    if current_tokens:
        entities.append(Entity(
            text=" ".join(current_tokens),
            label=current_label,
            start=pos - len(" ".join(current_tokens)),
            end=pos,
        ))
    return entities


def run_pipeline(
    input_source: Union[str, Path],
    kb: KnowledgeBase = None,
    model_path: str = MODEL_CHECKPOINT,
    save_output: str = None,
) -> List[dict]:
    """
    Full Module 1 pipeline:
      1. Load text (raw string or PDF)
      2. Run BioBERT NER
      3. Extract entity spans
      4. Link to KB (DrugBank / NCBI Gene / MeSH)
      5. Return structured JSON-ready list

    Args:
        input_source : raw text string OR path to a PDF file
        kb           : KnowledgeBase instance (defaults to sample KB)
        model_path   : HuggingFace model ID or local path
        save_output  : optional path to save results as JSON

    Returns:
        List of dicts with keys: text, label, linked_id, linked_name, score
    """

    # ── Step 1: Load text ──────────────────────────────────────────────────
    if str(input_source).endswith(".pdf"):
        print(f"[INFO] Extracting text from PDF: {input_source}")
        text = extract_text_from_pdf(str(input_source))
    else:
        text = str(input_source)
    print(f"[INFO] Input text length: {len(text)} characters")

    # ── Step 2: NER ────────────────────────────────────────────────────────
    print("[INFO] Running NER...")
    tokenizer, model = load_ner_model(model_path)
    token_label_pairs = ner_predict(text, tokenizer, model)

    # ── Step 3: Extract spans ──────────────────────────────────────────────
    entities = extract_entities_from_tokens(token_label_pairs)
    print(f"[INFO] Extracted {len(entities)} entity mentions")

    # ── Step 4: Entity linking ─────────────────────────────────────────────
    if kb is None:
        kb = build_sample_kb()
    linker   = EntityLinker(kb)
    entities = linker.link_batch(entities)

    # ── Step 5: Structure output ───────────────────────────────────────────
    results = [
        {
            "text":        e.text,
            "label":       e.label,
            "linked_id":   e.linked_id,
            "linked_name": e.linked_name,
            "score":       round(e.score, 4),
        }
        for e in entities
    ]

    if save_output:
        Path(save_output).parent.mkdir(parents=True, exist_ok=True)
        with open(save_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Results saved to {save_output}")

    # Print summary
    print("\n── Linked Entities ──────────────────────────────────")
    for r in results:
        print(f"  [{r['label']}] '{r['text']}' "
              f"→ {r['linked_id']} | {r['linked_name']} "
              f"(score: {r['score']})")
    return results


if __name__ == "__main__":
    sample_text = (
        "Imatinib is used to treat chronic myeloid leukemia. "
        "TP53 mutations are associated with lung cancer. "
        "Cisplatin targets KRAS in colorectal cancer patients."
    )
    results = run_pipeline(
        input_source=sample_text,
        save_output="data/processed/module1_output.json",
    )