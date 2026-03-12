"""
Module 2 - Knowledge Graph Builder
Stores extracted (head, relation, tail) triplets into Neo4j.
Supports dynamic updates as new biomedical data arrives.
"""

import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# ── Neo4j connection ──────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


@dataclass
class Triple:
    """A knowledge graph triple: (head_entity, relation, tail_entity)."""
    head:      str
    relation:  str
    tail:      str
    head_type: str = "Entity"   # Drug | Gene | Disease
    tail_type: str = "Entity"
    confidence: float = 1.0


class KnowledgeGraphBuilder:
    """
    Manages the Neo4j cancer knowledge graph.

    Node types  : Drug, Gene, Disease
    Relationships: TREATS, TARGETS, ASSOCIATED_WITH, INHIBITS, CAUSES
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        print(f"[INFO] Connected to Neo4j at {NEO4J_URI}")
        self._create_constraints()

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        """Create uniqueness constraints for each node type."""
        with self.driver.session() as session:
            for label in ["Drug", "Gene", "Disease"]:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                )
        print("[INFO] Neo4j constraints created")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_triple(self, triple: Triple):
        """Insert a single triple into the graph."""
        cypher = f"""
        MERGE (h:{triple.head_type} {{name: $head}})
        MERGE (t:{triple.tail_type} {{name: $tail}})
        MERGE (h)-[r:{triple.relation}]->(t)
        ON CREATE SET r.confidence = $confidence
        ON MATCH  SET r.confidence = $confidence
        """
        with self.driver.session() as session:
            session.run(cypher, {
                "head":       triple.head,
                "tail":       triple.tail,
                "confidence": triple.confidence,
            })

    def add_triples_batch(self, triples: List[Triple]):
        """Insert a batch of triples efficiently."""
        print(f"[INFO] Inserting {len(triples)} triples into Neo4j...")
        for triple in triples:
            self.add_triple(triple)
        print(f"[INFO] Done. Graph now contains {self.count_nodes()} nodes "
              f"and {self.count_relationships()} relationships.")

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """Execute a raw Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def get_entity_relationships(self, entity_name: str) -> List[Dict]:
        """Get all relationships for a given entity."""
        cypher = """
        MATCH (n {name: $name})-[r]-(m)
        RETURN n.name AS head, type(r) AS relation, m.name AS tail
        """
        return self.query(cypher, {"name": entity_name})

    def get_drugs_for_disease(self, disease: str) -> List[str]:
        """Find all drugs that treat a given disease."""
        cypher = """
        MATCH (d:Drug)-[:TREATS]->(dis:Disease {name: $disease})
        RETURN d.name AS drug
        """
        results = self.query(cypher, {"disease": disease})
        return [r["drug"] for r in results]

    def get_genes_for_disease(self, disease: str) -> List[str]:
        """Find all genes associated with a given disease."""
        cypher = """
        MATCH (g:Gene)-[:ASSOCIATED_WITH]->(d:Disease {name: $disease})
        RETURN g.name AS gene
        """
        results = self.query(cypher, {"disease": disease})
        return [r["gene"] for r in results]

    def count_nodes(self) -> int:
        result = self.query("MATCH (n) RETURN count(n) AS count")
        return result[0]["count"] if result else 0

    def count_relationships(self) -> int:
        result = self.query("MATCH ()-[r]->() RETURN count(r) AS count")
        return result[0]["count"] if result else 0

    # ── Sample data ───────────────────────────────────────────────────────────

    def load_sample_triples(self):
        """Load sample cancer knowledge triples for testing."""
        sample_triples = [
            Triple("Gefitinib",  "TREATS",          "Lung Cancer",       "Drug",    "Disease", 0.95),
            Triple("Cisplatin",  "TREATS",          "Lung Cancer",       "Drug",    "Disease", 0.92),
            Triple("Docetaxel",  "TREATS",          "Lung Cancer",       "Drug",    "Disease", 0.88),
            Triple("Gefitinib",  "TARGETS",         "EGFR",              "Drug",    "Gene",    0.97),
            Triple("Cisplatin",  "TARGETS",         "ALK",               "Drug",    "Gene",    0.85),
            Triple("Cisplatin",  "TARGETS",         "KRAS",              "Drug",    "Gene",    0.83),
            Triple("TP53",       "ASSOCIATED_WITH", "Lung Cancer",       "Gene",    "Disease", 0.91),
            Triple("KRAS",       "ASSOCIATED_WITH", "Colorectal Cancer", "Gene",    "Disease", 0.89),
            Triple("Gefitinib",  "INHIBITS",        "EGFR",              "Drug",    "Gene",    0.96),
            Triple("Imatinib",   "TREATS",          "Leukemia",          "Drug",    "Disease", 0.98),
            Triple("Imatinib",   "TARGETS",         "BCR-ABL",           "Drug",    "Gene",    0.99),
            Triple("BCR-ABL",    "CAUSES",          "Leukemia",          "Gene",    "Disease", 0.94),
        ]
        self.add_triples_batch(sample_triples)


if __name__ == "__main__":
    kg = KnowledgeGraphBuilder()
    kg.load_sample_triples()

    print("\n── Drugs for Lung Cancer ──")
    for drug in kg.get_drugs_for_disease("Lung Cancer"):
        print(f"  {drug}")

    print("\n── Genes for Lung Cancer ──")
    for gene in kg.get_genes_for_disease("Lung Cancer"):
        print(f"  {gene}")

    kg.close()