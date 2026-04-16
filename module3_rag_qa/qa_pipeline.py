"""
Module 3 - Full RAG QA Pipeline
Orchestrates: Question → Cypher → Neo4j → Context → Answer

Corrected version:
- better validation
- stronger debug logging
- safer fallback handling
- cleaner return structure
"""

from typing import Dict, Any
from .cypher_generator import generate_and_validate
from .retriever import GraphRetriever
from .answer_generator import AnswerGenerator, BiomedicalSummarizer


class MedKnowQA:
    """
    End-to-end RAG Question Answering system over the
    Cancer Knowledge Graph.

    Flow:
      1. User asks a natural language question
      2. Cypher is generated and validated
      3. Neo4j executes the query → subgraph retrieved
      4. Context is built from retrieved facts
      5. Answer generator produces grounded answer
      6. Answer + supporting evidence returned
    """

    def __init__(self, load_summarizer: bool = False, debug: bool = True):
        self.retriever = GraphRetriever()
        self.generator = AnswerGenerator()
        self.summarizer = BiomedicalSummarizer() if load_summarizer else None
        self.debug = debug

        if self.debug:
            print("[INFO] MedKnowQA system ready.")

    def _empty_result(
        self,
        question: str,
        answer: str,
        cypher: str = "",
        context: str = "",
        sources: list | None = None,
        raw_results: list | None = None,
        success: bool = False,
        error: str = "",
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "question": question,
            "cypher": cypher,
            "answer": answer,
            "sources": sources or [],
            "context": context,
            "raw_results": raw_results or [],
            "error": error,
        }

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a cancer-related question and get a grounded answer.

        Returns:
            {
                success: bool,
                question: str,
                cypher: str,
                answer: str,
                sources: list,
                context: str,
                raw_results: list,
                error: str
            }
        """
        question = (question or "").strip()

        if not question:
            return self._empty_result(
                question="",
                answer="Please enter a question.",
                error="Empty question.",
            )

        if self.debug:
            print("\n" + "─" * 70)
            print(f"[QUESTION] {question}")

        try:
            # Step 1: Generate + validate Cypher
            is_valid, cypher = generate_and_validate(question)

            if self.debug:
                print(f"[DEBUG] Generated Cypher: {cypher}")

            if not is_valid:
                return self._empty_result(
                    question=question,
                    cypher=cypher,
                    answer="Could not generate a valid query for this question.",
                    error="Cypher validation failed.",
                )

            # Step 2: Retrieve from Neo4j
            raw_results, context = self.retriever.retrieve_and_format(cypher, question)

            if self.debug:
                result_count = len(raw_results) if raw_results else 0
                print(f"[DEBUG] Raw result count: {result_count}")
                print(f"[DEBUG] Context preview: {context[:500] if context else 'EMPTY'}")

            if not raw_results:
                return self._empty_result(
                    question=question,
                    cypher=cypher,
                    answer="No relevant information found in the knowledge graph.",
                    context=context,
                    error="Neo4j returned no rows.",
                )

            # Step 3: Generate grounded answer
            result = self.generator.generate_with_sources(question, context, raw_results)

            # Defensive cleanup
            if not isinstance(result, dict):
                return self._empty_result(
                    question=question,
                    cypher=cypher,
                    answer="The answer generator returned an invalid response.",
                    context=context,
                    raw_results=raw_results,
                    error="Answer generator output was not a dict.",
                )

            result.setdefault("answer", "No answer could be generated.")
            result.setdefault("sources", [])
            result["success"] = True
            result["question"] = question
            result["cypher"] = cypher
            result["context"] = context
            result["raw_results"] = raw_results
            result["error"] = ""

            if self.debug:
                print(f"[ANSWER] {result['answer']}")
                if result["sources"]:
                    print("[SOURCES]")
                    for src in result["sources"]:
                        print(f"  • {src}")

            return result

        except Exception as e:
            if self.debug:
                print(f"[ERROR] QA pipeline failed: {e}")

            return self._empty_result(
                question=question,
                answer="The QA system encountered an internal error.",
                error=str(e),
            )

    def summarize_context(self, text: str) -> str:
        """
        Optional summarization utility.
        """
        if not text.strip():
            return "No text provided for summarization."

        if self.summarizer is None:
            return "Summarizer not loaded."

        try:
            summary = self.summarizer.summarize(text)
            return summary if summary else "No summary could be generated."
        except Exception as e:
            if self.debug:
                print(f"[ERROR] Summarization failed: {e}")
            return f"Summarization failed: {e}"

    def interactive(self):
        """
        Run an interactive QA session in the terminal.
        Type 'exit' to quit.
        """
        print("\n" + "=" * 70)
        print("              MedKnow-GEN Cancer KG QA System")
        print("=" * 70)
        print("Ask cancer-related questions. Type 'exit' to quit.\n")

        while True:
            question = input("Ask a cancer-related question: ").strip()

            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            if not question:
                print("Please enter a non-empty question.\n")
                continue

            result = self.ask(question)

            print("\nAnswer:", result["answer"])
            print("Cypher:", result["cypher"])

            if result["sources"]:
                print("Sources:")
                for src in result["sources"]:
                    print(f" - {src}")

            if result["error"]:
                print("Error:", result["error"])

            print()

    def close(self):
        try:
            self.retriever.close()
            if self.debug:
                print("[INFO] Retriever connection closed.")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to close retriever cleanly: {e}")


if __name__ == "__main__":
    qa = MedKnowQA(debug=True)
    try:
        qa.interactive()
    finally:
        qa.close()