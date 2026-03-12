"""
Module 3 - Graph Retriever
Executes validated Cypher queries against Neo4j,
retrieves relevant subgraphs, and formats context
for the answer generation step.
"""

import os
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class GraphRetriever:
    """
    Retrieves subgraphs from Neo4j based on validated Cypher queries.
    Formats results as natural language context for RAG generation.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        print(f"[INFO] GraphRetriever connected to Neo4j")

    def close(self):
        self.driver.close()

    def execute_cypher(self, cypher: str) -> List[Dict]:
        """Execute a Cypher query and return raw results."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                return [record.data() for record in result]
        except Exception as e:
            print(f"[ERROR] Cypher execution failed: {e}")
            return []

    def retrieve_subgraph(
        self, entity_name: str, depth: int = 1
    ) -> List[Dict]:
        """
        Retrieve the subgraph around an entity up to a given depth.
        Returns list of (head, relation, tail) dicts.
        """
        cypher = """
        MATCH (n {name: $name})-[r*1..%d]-(m)
        RETURN n.name AS head, type(r[0]) AS relation, m.name AS tail
        LIMIT 50
        """ % depth
        return self.execute_cypher_with_params(cypher, {"name": entity_name})

    def execute_cypher_with_params(
        self, cypher: str, params: Dict
    ) -> List[Dict]:
        """Execute a parameterized Cypher query."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher, params)
                return [record.data() for record in result]
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return []

    def format_as_context(
        self, results: List[Dict], question: str
    ) -> str:
        """
        Format raw Neo4j results as natural language context
        for the answer generator.
        """
        if not results:
            return "No relevant information found in the knowledge graph."

        context_lines = [
            f"From the knowledge graph I found the following facts:"
        ]
        for r in results:
            # Handle different result shapes
            if "drug" in r:
                context_lines.append(f"{r['drug']} treats the condition.")
            elif "gene" in r:
                context_lines.append(f"{r['gene']} is associated with the condition.")
            elif all(k in r for k in ["head", "relation", "tail"]):
                context_lines.append(
                    f"{r['head']} {r['relation'].lower().replace('_', ' ')} "
                    f"{r['tail']}."
                )
            else:
                # Generic: join all values
                context_lines.append("; ".join(str(v) for v in r.values()))

        return "\n".join(context_lines)

    def retrieve_and_format(
        self, cypher: str, question: str
    ) -> Tuple[List[Dict], str]:
        """
        Execute Cypher + format results as context string.
        Returns (raw_results, formatted_context).
        """
        results = self.execute_cypher(cypher)
        context = self.format_as_context(results, question)
        print(f"[INFO] Retrieved {len(results)} results from graph")
        return results, context