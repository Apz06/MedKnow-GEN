"""
Module 2 - Relation Extractor
Fine-tuned BioLinkBERT / SciBERT to classify semantic relationships
between biomedical entity pairs extracted from Module 1.

Input:  (entity1, entity2, sentence) triplets
Output: (head, relation, tail) triples → e.g. (Imatinib, TREATS, Leukemia)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import numpy as np


# ── Relation types (Drug-Gene-Disease) ───────────────────────────────────────
RELATIONS = [
    "TREATS",           # Drug → Disease
    "TARGETS",          # Drug → Gene
    "ASSOCIATED_WITH",  # Gene → Disease
    "INHIBITS",         # Drug → Gene
    "CAUSES",           # Gene → Disease
    "NO_RELATION",      # negative
]
REL2ID = {r: i for i, r in enumerate(RELATIONS)}
ID2REL = {i: r for r, i in REL2ID.items()}

MODEL_CHECKPOINT = "sultan/BioM-ELECTRA-Large-SQuAD2"
# Alternative: "allenai/scibert_scivocab_uncased"


class RelationDataset(Dataset):
    """
    Dataset for relation classification.
    Each sample: (entity1, entity2, sentence, label)
    Encodes as: [CLS] sent [SEP] e1 [SEP] e2 [SEP]
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, str, int]],
        tokenizer,
        max_length: int = 256,
    ):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e1, e2, sentence, label = self.samples[idx]
        encoding = self.tokenizer(
            sentence,
            f"{e1} [SEP] {e2}",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


class RelationClassifier(nn.Module):
    """
    Transformer + softmax classifier for relation extraction.
    Uses [CLS] token representation for classification.
    """

    def __init__(self, model_checkpoint: str, num_labels: int):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_checkpoint)
        hidden_size    = self.encoder.config.hidden_size
        self.dropout   = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)
        return logits


def build_sample_training_data() -> List[Tuple[str, str, str, int]]:
    """
    Sample training triplets for testing.
    Replace with BioRED / ChemProt / DDI corpus in production.
    Format: (entity1, entity2, sentence, relation_id)
    """
    return [
        ("Imatinib", "Leukemia",
         "Imatinib is a first-line treatment for chronic myeloid leukemia.",
         REL2ID["TREATS"]),
        ("Cisplatin", "KRAS",
         "Cisplatin has been shown to target KRAS in cancer cells.",
         REL2ID["TARGETS"]),
        ("TP53", "Lung Cancer",
         "TP53 mutations are strongly associated with lung cancer progression.",
         REL2ID["ASSOCIATED_WITH"]),
        ("Gefitinib", "EGFR",
         "Gefitinib inhibits EGFR signaling in non-small cell lung cancer.",
         REL2ID["INHIBITS"]),
        ("KRAS", "Colorectal Cancer",
         "Activating KRAS mutations cause colorectal cancer development.",
         REL2ID["CAUSES"]),
    ]


def train_relation_extractor(
    samples: List[Tuple],
    output_dir: str = "models/relation_extractor",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    """Train the relation classifier."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model     = RelationClassifier(MODEL_CHECKPOINT, num_labels=len(RELATIONS))
    model.to(device)

    dataset    = RelationDataset(samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss()

    print(f"[INFO] Training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(dim=-1) == labels).sum().item()

        acc = correct / len(dataset)
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss:.4f} | Acc: {acc:.4f}")

    torch.save(model.state_dict(), f"{output_dir}/relation_model.pt")
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved to {output_dir}")
    return model, tokenizer


def predict_relation(
    e1: str, e2: str, sentence: str,
    model: RelationClassifier,
    tokenizer,
    device: str = "cpu",
) -> Tuple[str, float]:
    """
    Predict relation between two entities given a sentence context.
    Returns (relation_label, confidence_score).
    """
    model.eval()
    encoding = tokenizer(
        sentence, f"{e1} [SEP] {e2}",
        truncation=True, padding="max_length",
        max_length=256, return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        )
    probs      = torch.softmax(logits, dim=-1)[0]
    pred_id    = probs.argmax().item()
    confidence = probs[pred_id].item()
    return ID2REL[pred_id], confidence


if __name__ == "__main__":
    samples = build_sample_training_data()
    model, tokenizer = train_relation_extractor(samples)

    # Test prediction
    rel, conf = predict_relation(
        "Gefitinib", "Lung Cancer",
        "Gefitinib is used to treat non-small cell lung cancer.",
        model, tokenizer,
    )
    print(f"\nPredicted: {rel} (confidence: {conf:.3f})")