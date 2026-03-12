"""
Module 3 - Full RAG QA Pipeline
Orchestrates: Question → Cypher → Neo4j → Context → Answer

This is the main entry point for the QA system.
"""

from .cypher_generator import generate_and_validate
from .retriever        import GraphRetriever
from .answer_generator import AnswerGenerator, BiomedicalSummarizer
from typing import Dict, Optional


class MedKnowQA:
    """
    End-to-end RAG Question Answering system over the
    Cancer Knowledge Graph.

    Flow:
      1. User asks a natural language question
      2. LLM converts it to a Cypher query
      3. Cypher is validated and sanitized
      4. Neo4j executes the query → subgraph retrieved
      5. FLAN-T5 generates a grounded answer from context
      6. Answer + supporting evidence returned
    """

    def __init__(self, load_summarizer: bool = False):
        self.retriever   = GraphRetriever()
        self.generator   = AnswerGenerator()
        self.summarizer  = BiomedicalSummarizer() if load_summarizer else None
        print("[INFO] MedKnowQA system ready.")

    def ask(self, question: str) -> Dict:
        """
        Ask a cancer-related question and get a grounded answer.

        Returns dict with:
          - question   : original question
          - cypher     : generated Cypher query
          - answer     : generated answer
          - sources    : supporting KG facts
          - context    : raw formatted context
        """
        print(f"\n── Question: {question}")

        # Step 1: Generate + validate Cypher
        is_valid, cypher = generate_and_validate(question)
        if not is_valid:
            return {
                "question": question,
                "cypher":   cypher,
                "answer":   "Could not generate a valid query for this question.",
                "sources":  [],
                "context":  "",
            }

        # Step 2: Retrieve from Neo4j
        raw_results, context = self.retriever.retrieve_and_format(
            cypher, question
        )

        if not raw_results:
            return {
                "question": question,
                "cypher":   cypher,
                "answer":   "No relevant information found in the knowledge graph.",
                "sources":  [],
                "context":  context,
            }

        # Step 3: Generate grounded answer
        result = self.generator.generate_with_sources(
            question, context, raw_results
        )
        result["cypher"] = cypher

        print(f"── Answer: {result['answer']}")
        if result["sources"]:
            print("── Sources:")
            for s in result["sources"]:
                print(f"   • {s}")

        return result

    def interactive(self):
        """
        Run an interactive QA session in the terminal.
        Type 'exit' to quit.
        """
        print("\n" + "="*60)
        print("  MedKnow-GEN Cancer Knowledge Graph QA System")
        print("="*60)
        print("Ask cancer-related questions (type 'exit' to quit)\n")

        while True:
            question = input("Ask a cancer-related question: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue

            result = self.ask(question)
            print(f"\nAnswer: {result['answer']}\n")

    def close(self):
        self.retriever.close()


if __name__ == "__main__":
    qa = MedKnowQA()
    qa.interactive()
    qa.close()