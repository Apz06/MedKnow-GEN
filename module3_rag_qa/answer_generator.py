"""
Module 3 - Answer Generator
Generates grounded, evidence-backed answers using:
  - FLAN-T5 fine-tuned for biomedical QA
  - Retrieved knowledge graph context (RAG)
  - Hybrid extractive-abstractive summarization
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline as hf_pipeline,
)
from typing import List, Dict, Optional


# ── Model options ─────────────────────────────────────────────────────────────
QA_MODEL          = "google/flan-t5-base"
SUMMARIZER_MODEL  = "facebook/bart-large-cnn"
MAX_INPUT_LENGTH  = 512
MAX_OUTPUT_LENGTH = 200


class AnswerGenerator:
    """
    RAG answer generator: combines KG context with
    FLAN-T5 to produce grounded biomedical answers.
    """

    def __init__(self, model_name: str = QA_MODEL):
        print(f"[INFO] Loading answer generator: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Model loaded on {self.device}")

    def generate(
        self,
        question:    str,
        context:     str,
        max_length:  int = MAX_OUTPUT_LENGTH,
        num_beams:   int = 4,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate an answer grounded in the KG context.

        Prompt format:
          Answer the biomedical question based only on the provided context.
          Context: <kg_facts>
          Question: <question>
          Answer:
        """
        prompt = (
            f"Answer the biomedical question based only on the context.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        answer = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        return answer.strip()

    def generate_with_sources(
        self,
        question: str,
        context:  str,
        raw_results: List[Dict],
    ) -> Dict:
        """
        Generate answer + attach source triples as evidence.
        Returns dict with answer and supporting facts.
        """
        answer  = self.generate(question, context)
        sources = [
            "; ".join(f"{k}: {v}" for k, v in r.items())
            for r in raw_results[:5]   # top 5 supporting facts
        ]
        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "context":  context,
        }


class BiomedicalSummarizer:
    """
    Hybrid extractive-abstractive summarizer for scientific findings.
    Uses BART for abstractive summarization.
    """

    def __init__(self):
        print(f"[INFO] Loading summarizer: {SUMMARIZER_MODEL}")
        self.summarizer = hf_pipeline(
            "summarization",
            model=SUMMARIZER_MODEL,
            device=0 if torch.cuda.is_available() else -1,
        )

    def summarize(
        self,
        text:       str,
        max_length: int = 150,
        min_length: int = 40,
    ) -> str:
        """Generate an abstractive summary of biomedical text."""
        if len(text.split()) < 50:
            return text   # too short to summarize

        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        return result[0]["summary_text"]

    def extractive_summary(
        self, text: str, top_n: int = 3
    ) -> str:
        """
        Simple extractive summary: return the top N most
        informative sentences (by length heuristic).
        """
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        # Score by length (longer = more informative as a heuristic)
        scored    = sorted(sentences, key=len, reverse=True)
        return ". ".join(scored[:top_n]) + "."