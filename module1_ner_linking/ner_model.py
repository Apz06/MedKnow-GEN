"""
Module 1 - NER Model
Fine-tunes BioBERT / PubMedBERT on BC5CDR corpus for
biomedical Named Entity Recognition (genes, drugs, diseases).
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report

# ── Label schema (BC5CDR: Chemical + Disease NER) ────────────────────────────
LABEL_LIST = [
    "O",
    "B-Chemical", "I-Chemical",
    "B-Disease",  "I-Disease",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# ── Model options ─────────────────────────────────────────────────────────────
# "dmis-lab/biobert-base-cased-v1.2"
# "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MODEL_CHECKPOINT = "dmis-lab/biobert-base-cased-v1.2"


def load_bc5cdr_dataset():
    """Load BC5CDR corpus from HuggingFace datasets."""
    print("[INFO] Loading BC5CDR dataset...")
    dataset = load_dataset("tner/bc5cdr")
    print(f"[INFO] Train: {len(dataset['train'])} | "
          f"Val: {len(dataset['validation'])} | "
          f"Test: {len(dataset['test'])}")
    return dataset


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize inputs and align NER labels with subword tokens.
    Uses -100 for special tokens (ignored by loss).
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=512,
    )
    all_labels = []
    for i, labels in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)          # special token
            elif word_id != prev_word_id:
                aligned.append(labels[word_id])  # first subword
            else:
                aligned.append(-100)          # continuation subword
            prev_word_id = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_pred):
    """Compute token-level precision, recall, F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(ID2LABEL[l])
                true_preds.append(ID2LABEL[p])

    report = classification_report(
        true_labels, true_preds, output_dict=True, zero_division=0
    )
    return {
        "precision": report["weighted avg"]["precision"],
        "recall":    report["weighted avg"]["recall"],
        "f1":        report["weighted avg"]["f1-score"],
    }


def build_model(num_labels: int):
    """Load pretrained BioBERT with token classification head."""
    print(f"[INFO] Loading model: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return tokenizer, model


def train(output_dir: str = "models/ner_biobert"):
    """Full training pipeline."""
    dataset  = load_bc5cdr_dataset()
    tokenizer, model = build_model(num_labels=len(LABEL_LIST))

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
    )

    # Training config
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="logs/ner_training",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved to {output_dir}")
    return trainer, tokenizer, model


def predict(text: str, tokenizer, model, device: str = "cpu"):
    """
    Run NER inference on a single text string.
    Returns list of (token, label) tuples.
    """
    model.eval()
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits      = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()
    tokens      = tokenizer.convert_ids_to_tokens(
                      inputs["input_ids"][0].tolist()
                  )

    results = [
        (tok, ID2LABEL[pred])
        for tok, pred in zip(tokens, predictions)
        if tok not in ("[CLS]", "[SEP]", "[PAD]")
    ]
    return results


if __name__ == "__main__":
    train()