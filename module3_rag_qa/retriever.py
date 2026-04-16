"""
Module 3 - Graph Retriever
Executes validated Cypher queries against Neo4j,
retrieves relevant subgraphs, and formats context
for the answer generation step.
"""

import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class GraphRetriever:
    """
    Retrieves data from Neo4j based on validated Cypher queries.
    Formats results as readable context for downstream answer generation.
    """

    def __init__(self, debug: bool = True):
        self.debug = debug
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        self._verify_connection()

    def _verify_connection(self):
        """Verify Neo4j connection on startup."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1 AS ok").single()
            if self.debug:
                print(f"[INFO] GraphRetriever connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            print(f"[ERROR] Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close Neo4j driver cleanly."""
        try:
            self.driver.close()
            if self.debug:
                print("[INFO] Neo4j driver closed.")
        except Exception as e:
            print(f"[WARN] Failed to close Neo4j driver cleanly: {e}")

    def execute_cypher(self, cypher: str) -> List[Dict]:
        """Execute a Cypher query and return raw results."""
        try:
            if self.debug:
                print(f"[DEBUG] Executing Cypher:\n{cypher}")

            with self.driver.session() as session:
                result = session.run(cypher)
                rows = [record.data() for record in result]

            if self.debug:
                print(f"[DEBUG] Rows returned: {len(rows)}")
                if rows:
                    print(f"[DEBUG] First row: {rows[0]}")

            return rows

        except Exception as e:
            print(f"[ERROR] Cypher execution failed: {e}")
            return []

    def execute_cypher_with_params(self, cypher: str, params: Dict) -> List[Dict]:
        """Execute a parameterized Cypher query."""
        try:
            if self.debug:
                print(f"[DEBUG] Executing parameterized Cypher:\n{cypher}")
                print(f"[DEBUG] Params: {params}")

            with self.driver.session() as session:
                result = session.run(cypher, params)
                rows = [record.data() for record in result]

            if self.debug:
                print(f"[DEBUG] Rows returned: {len(rows)}")
                if rows:
                    print(f"[DEBUG] First row: {rows[0]}")

            return rows

        except Exception as e:
            print(f"[ERROR] Parameterized Cypher execution failed: {e}")
            return []

    def retrieve_subgraph(self, entity_name: str, depth: int = 1) -> List[Dict]:
        """
        Retrieve the subgraph around an entity up to a given depth.
        Returns list of dicts with head, relation, tail.
        """
        cypher = f"""
        MATCH (n {{name: $name}})-[r*1..{depth}]-(m)
        RETURN n.name AS head, type(r[0]) AS relation, m.name AS tail
        LIMIT 50
        """
        return self.execute_cypher_with_params(cypher, {"name": entity_name})

    def format_as_context(self, results: List[Dict], question: str) -> str:
        """
        Format raw Neo4j results as readable natural language context
        for the answer generator.
        """
        if not results:
            return "No relevant information found in the knowledge graph."

        question = (question or "").strip()
        context_lines = ["From the knowledge graph I found the following facts:"]

        for r in results:
            # Drug answers
            if "drug" in r:
                context_lines.append(f"Drug: {r['drug']}")

            # Gene answers
            elif "gene" in r:
                context_lines.append(f"Gene: {r['gene']}")

            # Disease answers
            elif "disease" in r:
                context_lines.append(f"Disease: {r['disease']}")

            # Rich graph triplets
            elif all(k in r for k in ["head", "relation", "tail"]):
                relation = str(r["relation"]).lower().replace("_", " ")
                context_lines.append(f"{r['head']} {relation} {r['tail']}.")

            # Generic row fallback
            else:
                parts = []
                for k, v in r.items():
                    parts.append(f"{k}: {v}")
                context_lines.append("; ".join(parts))

        return "\n".join(context_lines)

    def retrieve_and_format(self, cypher: str, question: str) -> Tuple[List[Dict], str]:
        """
        Execute Cypher and format results as a context string.
        Returns:
            (raw_results, formatted_context)
        """
        results = self.execute_cypher(cypher)
        context = self.format_as_context(results, question)

        if self.debug:
            print(f"[INFO] Retrieved {len(results)} results from graph")
            print(f"[DEBUG] Context preview:\n{context[:500]}")

        return results, context


if __name__ == "__main__":
    retriever = GraphRetriever(debug=True)

    test_query = """
    MATCH (drug:Drug)-[:TREATS]->(disease:Disease)
    WHERE toLower(disease.name) = toLower("Lung Cancer")
    RETURN drug.name AS drug
    LIMIT 5
    """

    results, context = retriever.retrieve_and_format(
        test_query,
        "Which drug treats Lung Cancer?"
    )

    print("\nRAW RESULTS:")
    print(results)

    print("\nFORMATTED CONTEXT:")
    print(context)

    retriever.close()