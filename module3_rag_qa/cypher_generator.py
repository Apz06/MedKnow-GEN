"""
Module 3 - Cypher Generator
Converts natural language questions into Neo4j Cypher queries
using LLMs (Llama3 via Ollama locally, or GPT as fallback).
Includes validation and sanitization before execution.
"""

import os
import re
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# ── LLM backend: ollama (local) or openai ────────────────────────────────────
USE_OLLAMA = True   # set False to use OpenAI GPT instead
OLLAMA_MODEL  = "llama3"
OPENAI_MODEL  = "gpt-3.5-turbo"

CYPHER_PROMPT_TEMPLATE = """
You are an expert Neo4j Cypher query generator for a cancer biomedical 
knowledge graph.

The graph has these node types: Drug, Gene, Disease
The graph has these relationships: TREATS, TARGETS, ASSOCIATED_WITH, 
INHIBITS, CAUSES

Rules:
- Only generate READ queries (MATCH, RETURN) — never DELETE or CREATE
- Always use node labels (e.g. :Drug, :Disease)
- Return meaningful property names
- Keep queries simple and precise

Examples:
Q: Which drugs treat Lung Cancer?
A: MATCH (d:Drug)-[:TREATS]->(dis:Disease {{name: "Lung Cancer"}}) 
   RETURN d.name AS drug

Q: Which genes are associated with Breast Cancer?
A: MATCH (g:Gene)-[:ASSOCIATED_WITH]->(d:Disease {{name: "Breast Cancer"}}) 
   RETURN g.name AS gene

Q: What does Cisplatin target?
A: MATCH (d:Drug {{name: "Cisplatin"}})-[:TARGETS]->(g:Gene) 
   RETURN g.name AS gene

Now generate a Cypher query for:
Q: {question}
A:"""


def generate_cypher_ollama(question: str) -> str:
    """Generate Cypher using local Llama3 via Ollama."""
    try:
        import ollama
        prompt   = CYPHER_PROMPT_TEMPLATE.format(question=question)
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[WARN] Ollama failed: {e}. Falling back to rule-based.")
        return _rule_based_cypher(question)


def generate_cypher_openai(question: str) -> str:
    """Generate Cypher using OpenAI GPT."""
    try:
        from openai import OpenAI
        client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt   = CYPHER_PROMPT_TEMPLATE.format(question=question)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] OpenAI failed: {e}. Falling back to rule-based.")
        return _rule_based_cypher(question)


def _rule_based_cypher(question: str) -> str:
    """
    Fallback rule-based Cypher generator.
    Handles the most common cancer KG query patterns.
    """
    q = question.lower()

    # Pattern: drug → disease
    if any(w in q for w in ["treat", "drug for", "medication for"]):
        disease = _extract_entity(question, ["cancer", "leukemia",
                                             "tumor", "carcinoma"])
        if disease:
            return (f'MATCH (d:Drug)-[:TREATS]->(dis:Disease '
                    f'{{name: "{disease}"}}) RETURN d.name AS drug')

    # Pattern: gene → disease
    if any(w in q for w in ["gene", "associated", "linked to"]):
        disease = _extract_entity(question, ["cancer", "leukemia",
                                             "tumor", "carcinoma"])
        if disease:
            return (f'MATCH (g:Gene)-[:ASSOCIATED_WITH]->(d:Disease '
                    f'{{name: "{disease}"}}) RETURN g.name AS gene')

    # Pattern: drug → gene target
    if any(w in q for w in ["target", "inhibit"]):
        drug = _extract_capitalized(question)
        if drug:
            return (f'MATCH (d:Drug {{name: "{drug}"}})-[:TARGETS]->(g:Gene) '
                    f'RETURN g.name AS gene')

    # Generic fallback
    return "MATCH (n) RETURN n.name AS name LIMIT 10"


def _extract_entity(text: str, keywords: list) -> Optional[str]:
    """Extract disease entity by looking for capitalized words near keywords."""
    for kw in keywords:
        pattern = rf'([A-Z][a-zA-Z\s]*{kw}[a-zA-Z\s]*)'
        match   = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_capitalized(text: str) -> Optional[str]:
    """Extract the first capitalized word (likely a drug/gene name)."""
    match = re.search(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)?)\b', text)
    return match.group(1) if match else None


def sanitize_cypher(cypher: str) -> Tuple[bool, str]:
    """
    Validate and sanitize generated Cypher.
    Blocks write operations, checks basic syntax.
    Returns (is_valid, cleaned_cypher).
    """
    cypher = cypher.strip()

    # Extract just the Cypher if wrapped in markdown code block
    code_match = re.search(r'```(?:cypher)?\s*(.*?)\s*```', cypher, re.DOTALL)
    if code_match:
        cypher = code_match.group(1).strip()

    # Extract first line that looks like Cypher
    lines = [l.strip() for l in cypher.split('\n') if l.strip()]
    cypher_lines = [l for l in lines if any(
        l.upper().startswith(kw)
        for kw in ["MATCH", "WITH", "CALL", "RETURN"]
    )]
    if cypher_lines:
        cypher = " ".join(cypher_lines)

    # Block dangerous operations
    blocked = ["DELETE", "DETACH", "REMOVE", "SET", "CREATE",
               "MERGE", "DROP", "LOAD CSV"]
    for op in blocked:
        if op in cypher.upper():
            return False, f"[BLOCKED] Query contains forbidden operation: {op}"

    # Must start with MATCH or WITH
    if not any(cypher.upper().startswith(kw) for kw in ["MATCH", "WITH"]):
        return False, "[INVALID] Query must start with MATCH or WITH"

    return True, cypher


def generate_and_validate(question: str) -> Tuple[bool, str]:
    """
    Full pipeline: question → Cypher → validate → return.
    Returns (is_valid, cypher_or_error_message).
    """
    if USE_OLLAMA:
        raw_cypher = generate_cypher_ollama(question)
    else:
        raw_cypher = generate_cypher_openai(question)

    is_valid, cypher = sanitize_cypher(raw_cypher)
    if not is_valid:
        print(f"[WARN] Invalid Cypher: {cypher}")
        # Fallback to rule-based
        cypher   = _rule_based_cypher(question)
        is_valid = True

    print(f"[INFO] Generated Cypher: {cypher}")
    return is_valid, cypher


if __name__ == "__main__":
    test_questions = [
        "Which drug treats Lung Cancer?",
        "Which gene is linked to Colorectal Cancer?",
        "What does Gefitinib target?",
        "Which drug targets KRAS?",
    ]
    for q in test_questions:
        valid, cypher = generate_and_validate(q)
        print(f"\nQ: {q}")
        print(f"Cypher: {cypher}")
        print(f"Valid: {valid}")