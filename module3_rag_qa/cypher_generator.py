"""
Module 3 - Corrected Cypher Generator
Converts natural language biomedical questions into Neo4j Cypher queries.

Design choice:
- Prefer deterministic rule-based templates for common KG questions
- Use LLM only as fallback for unsupported questions
- Validate and sanitize all generated Cypher before execution
"""

import os
import re
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

USE_OLLAMA = True          # Set False to use OpenAI instead
USE_LLM_FALLBACK = True    # If rule-based generation fails, try LLM

OLLAMA_MODEL = "llama3"
OPENAI_MODEL = "gpt-3.5-turbo"

CYPHER_PROMPT_TEMPLATE = """
You are an expert Neo4j Cypher query generator for a biomedical cancer
knowledge graph.

Schema:
- Node labels: Drug, Gene, Disease
- Relationships: TREATS, TARGETS, ASSOCIATED_WITH, INHIBITS, CAUSES

Rules:
- Generate ONLY read queries
- Use MATCH / WHERE / RETURN only
- Never generate CREATE, MERGE, DELETE, SET, REMOVE, DROP
- Always use correct node labels
- Keep queries simple and specific
- Use exact relationship names from schema
- Return meaningful aliases like drug, gene, disease

Examples:

Q: Which drugs treat Lung Cancer?
A:
MATCH (d:Drug)-[:TREATS]->(dis:Disease)
WHERE toLower(dis.name) = toLower("Lung Cancer")
RETURN d.name AS drug
LIMIT 10

Q: Which genes are associated with Breast Cancer?
A:
MATCH (g:Gene)-[:ASSOCIATED_WITH]->(d:Disease)
WHERE toLower(d.name) = toLower("Breast Cancer")
RETURN g.name AS gene
LIMIT 10

Q: What does Cisplatin target?
A:
MATCH (d:Drug)-[:TARGETS]->(g:Gene)
WHERE toLower(d.name) = toLower("Cisplatin")
RETURN g.name AS gene
LIMIT 10

Q: Which drug targets EGFR?
A:
MATCH (d:Drug)-[:TARGETS]->(g:Gene)
WHERE toUpper(g.name) = "EGFR"
RETURN d.name AS drug
LIMIT 10

Now generate a Cypher query for:

Q: {question}
A:
""".strip()


# -----------------------------------------------------------------------------
# LLM-based generation
# -----------------------------------------------------------------------------

def generate_cypher_ollama(question: str) -> str:
    """Generate Cypher using local Ollama."""
    try:
        import ollama

        prompt = CYPHER_PROMPT_TEMPLATE.format(question=question)
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()

    except Exception as e:
        print(f"[WARN] Ollama generation failed: {e}")
        return "[UNSUPPORTED]"


def generate_cypher_openai(question: str) -> str:
    """Generate Cypher using OpenAI."""
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[WARN] OPENAI_API_KEY not found.")
            return "[UNSUPPORTED]"

        client = OpenAI(api_key=api_key)
        prompt = CYPHER_PROMPT_TEMPLATE.format(question=question)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[WARN] OpenAI generation failed: {e}")
        return "[UNSUPPORTED]"


# -----------------------------------------------------------------------------
# Rule-based generation
# -----------------------------------------------------------------------------

def _rule_based_cypher(question: str) -> str:
    """
    Deterministic query templates for supported question patterns.
    Returns:
        - valid Cypher string
        - or '[UNSUPPORTED]' if pattern not matched
    """
    q = (question or "").strip().rstrip("?")
    if not q:
        return "[UNSUPPORTED]"

    q_lower = q.lower()

    # -------------------------------------------------------------------------
    # Which drug treats <disease>
    # Example: Which drug treats Lung Cancer?
    # -------------------------------------------------------------------------
    m = re.match(r"which drug treats (.+)", q_lower)
    if m:
        disease = _normalize_title_entity(m.group(1))
        return f'''
MATCH (drug:Drug)-[:TREATS]->(disease:Disease)
WHERE toLower(disease.name) = toLower("{disease}")
RETURN drug.name AS drug
LIMIT 10
'''.strip()

    m = re.match(r"which drugs treat (.+)", q_lower)
    if m:
        disease = _normalize_title_entity(m.group(1))
        return f'''
MATCH (drug:Drug)-[:TREATS]->(disease:Disease)
WHERE toLower(disease.name) = toLower("{disease}")
RETURN drug.name AS drug
LIMIT 10
'''.strip()

    # -------------------------------------------------------------------------
    # Which gene is linked to <disease>
    # Which genes are associated with <disease>
    # -------------------------------------------------------------------------
    m = re.match(r"which gene is linked to (.+)", q_lower)
    if m:
        disease = _normalize_title_entity(m.group(1))
        return f'''
MATCH (gene:Gene)-[:ASSOCIATED_WITH]->(disease:Disease)
WHERE toLower(disease.name) = toLower("{disease}")
RETURN gene.name AS gene
LIMIT 10
'''.strip()

    m = re.match(r"which genes are linked to (.+)", q_lower)
    if m:
        disease = _normalize_title_entity(m.group(1))
        return f'''
MATCH (gene:Gene)-[:ASSOCIATED_WITH]->(disease:Disease)
WHERE toLower(disease.name) = toLower("{disease}")
RETURN gene.name AS gene
LIMIT 10
'''.strip()

    m = re.match(r"which gene is associated with (.+)", q_lower)
    if m:
        disease = _normalize_title_entity(m.group(1))
        return f'''
MATCH (gene:Gene)-[:ASSOCIATED_WITH]->(disease:Disease)
WHERE toLower(disease.name) = toLower("{disease}")
RETURN gene.name AS gene
LIMIT 10
'''.strip()

    m = re.match(r"which genes are associated with (.+)", q_lower)
    if m:
        disease = _normalize_title_entity(m.group(1))
        return f'''
MATCH (gene:Gene)-[:ASSOCIATED_WITH]->(disease:Disease)
WHERE toLower(disease.name) = toLower("{disease}")
RETURN gene.name AS gene
LIMIT 10
'''.strip()

    # -------------------------------------------------------------------------
    # What does <drug> target
    # Example: What does Cisplatin target?
    # -------------------------------------------------------------------------
    m = re.match(r"what does (.+) target", q_lower)
    if m:
        drug = _normalize_title_entity(m.group(1))
        return f'''
MATCH (drug:Drug)-[:TARGETS]->(gene:Gene)
WHERE toLower(drug.name) = toLower("{drug}")
RETURN gene.name AS gene
LIMIT 10
'''.strip()

    # -------------------------------------------------------------------------
    # Which drug targets <gene>
    # Example: Which drug targets EGFR?
    # -------------------------------------------------------------------------
    m = re.match(r"which drug targets (.+)", q_lower)
    if m:
        gene = _normalize_gene_entity(m.group(1))
        return f'''
MATCH (drug:Drug)-[:TARGETS]->(gene:Gene)
WHERE toUpper(gene.name) = "{gene}"
RETURN drug.name AS drug
LIMIT 10
'''.strip()

    m = re.match(r"which drugs target (.+)", q_lower)
    if m:
        gene = _normalize_gene_entity(m.group(1))
        return f'''
MATCH (drug:Drug)-[:TARGETS]->(gene:Gene)
WHERE toUpper(gene.name) = "{gene}"
RETURN drug.name AS drug
LIMIT 10
'''.strip()

    # -------------------------------------------------------------------------
    # Explore disease by drug
    # Example: Which disease is associated with Carbamazepine?
    # Your earlier demo used this kind of question.
    # -------------------------------------------------------------------------
    m = re.match(r"which disease is associated with (.+)", q_lower)
    if m:
        drug = _normalize_title_entity(m.group(1))
        return f'''
MATCH (drug:Drug)-[:TREATS|ASSOCIATED_WITH]->(disease:Disease)
WHERE toLower(drug.name) = toLower("{drug}")
RETURN disease.name AS disease
LIMIT 10
'''.strip()

    # -------------------------------------------------------------------------
    # Unsupported question type
    # -------------------------------------------------------------------------
    return "[UNSUPPORTED]"


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------

def _normalize_title_entity(text: str) -> str:
    """
    Normalize disease/drug strings like:
    'lung cancer' -> 'Lung Cancer'
    'cisplatin' -> 'Cisplatin'
    """
    text = _clean_entity(text)
    return " ".join(word.capitalize() for word in text.split())


def _normalize_gene_entity(text: str) -> str:
    """
    Normalize gene-like strings:
    'egfr' -> 'EGFR'
    'tp53' -> 'TP53'
    """
    text = _clean_entity(text)
    return text.upper()


def _clean_entity(text: str) -> str:
    """Basic cleanup for extracted entity text."""
    text = text.strip()
    text = text.strip("?.,:;!()[]{}")
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------------------------------------------------------
# Sanitization / validation
# -----------------------------------------------------------------------------

def sanitize_cypher(cypher: str) -> Tuple[bool, str]:
    """
    Validate and sanitize generated Cypher.
    Returns (is_valid, cleaned_cypher_or_error).
    """
    if not cypher or cypher == "[UNSUPPORTED]":
        return False, "Unsupported question type for current graph templates."

    cypher = cypher.strip()

    # Extract Cypher from markdown blocks if present
    code_match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", cypher, re.DOTALL | re.IGNORECASE)
    if code_match:
        cypher = code_match.group(1).strip()

    # Keep only meaningful Cypher lines
    lines = [line.strip() for line in cypher.splitlines() if line.strip()]
    cypher_lines = [
        line for line in lines
        if line.upper().startswith(("MATCH", "OPTIONAL MATCH", "WHERE", "WITH", "RETURN", "LIMIT"))
    ]
    if cypher_lines:
        cypher = " ".join(cypher_lines)

    # Block dangerous operations
    blocked_ops = [
        "DELETE", "DETACH", "REMOVE", "SET", "CREATE", "MERGE",
        "DROP", "LOAD CSV", "CALL DBMS", "FOREACH"
    ]
    cypher_upper = cypher.upper()
    for op in blocked_ops:
        if op in cypher_upper:
            return False, f"[BLOCKED] Query contains forbidden operation: {op}"

    # Must start with MATCH or OPTIONAL MATCH
    if not (cypher_upper.startswith("MATCH") or cypher_upper.startswith("OPTIONAL MATCH")):
        return False, "[INVALID] Query must start with MATCH or OPTIONAL MATCH"

    # Must contain RETURN
    if "RETURN" not in cypher_upper:
        return False, "[INVALID] Query must contain RETURN"

    return True, cypher


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def generate_and_validate(question: str) -> Tuple[bool, str]:
    """
    Full pipeline:
      1. Try rule-based query generation first
      2. If unsupported, optionally try LLM fallback
      3. Sanitize and validate
    Returns:
      (is_valid, cypher_or_error_message)
    """
    # Prefer deterministic rules first
    raw_cypher = _rule_based_cypher(question)

    # Only use LLM if rules cannot handle the question
    if raw_cypher == "[UNSUPPORTED]" and USE_LLM_FALLBACK:
        if USE_OLLAMA:
            raw_cypher = generate_cypher_ollama(question)
        else:
            raw_cypher = generate_cypher_openai(question)

    is_valid, cypher = sanitize_cypher(raw_cypher)

    if not is_valid:
        print(f"[WARN] Cypher generation failed: {cypher}")
        return False, cypher

    print(f"[INFO] Generated Cypher: {cypher}")
    return True, cypher


# -----------------------------------------------------------------------------
# Local test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        "Which drug treats Lung Cancer?",
        "Which gene is linked to Colorectal Cancer?",
        "What does Cisplatin target?",
        "Which drug targets EGFR?",
        "Which disease is associated with Carbamazepine?",
        "Tell me something random",
    ]

    for q in test_questions:
        print("\n" + "=" * 80)
        print("Q:", q)
        valid, cypher = generate_and_validate(q)
        print("Valid:", valid)
        print("Cypher:", cypher)