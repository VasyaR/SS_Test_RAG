"""
LLM integration for article Q&A using Groq API.

Implements ArticleQABot class that generates answers from retrieved article context.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
from qdrant_client.models import Filter

from .database import MultimodalDB
from .prompt import PROMPT
from .retriever import MultimodalRetriever, build_filter

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class ArticleQABot:
    """Article Q&A bot using multimodal retrieval and Groq LLM."""

    def __init__(self, retriever: MultimodalRetriever, prompt: str = PROMPT):
        """
        Initialize Article Q&A bot.

        Args:
            retriever: MultimodalRetriever instance
            prompt: System prompt for LLM
        """
        self.retriever = retriever
        self.prompt = prompt

    def answer_question(
        self,
        query: str,
        n_results: int = 5,
        where: Filter | None = None,
        use_hybrid: bool = True
    ) -> dict:
        """
        Answer a question using retrieved article context.

        Args:
            query: User question
            n_results: Number of text chunks to retrieve
            where: Optional Qdrant filter for metadata filtering
            use_hybrid: Use hybrid BM25+semantic retrieval

        Returns:
            Dictionary with answer, context, sources, and images
        """
        # Retrieve multimodal results
        results = self.retriever.retrieve_multimodal(
            query=query, n_text=n_results, n_images=3, where=where
        )

        # Build context and generate answer
        context = self._build_context(results["text_results"])
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        response = completion(model="groq/llama-3.3-70b-versatile", messages=messages)

        return {
            "answer": response.choices[0].message.content,
            "context": context,
            "sources": self._extract_sources(results["text_results"]),
            "images": results["image_results"]
        }

    def _build_context(self, text_results: list[dict]) -> str:
        """Build context string from retrieved text chunks."""
        parts = []
        for result in text_results:
            meta = result["metadata"]
            date = meta.get("article_date", "Unknown date")[:10]
            parts.append(f"[{meta['article_title']}] ({date})\n{result['text']}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, text_results: list[dict]) -> list[dict]:
        """Extract unique article sources from results."""
        seen, sources = set(), []
        for result in text_results:
            meta = result["metadata"]
            if meta["article_id"] not in seen:
                seen.add(meta["article_id"])
                sources.append({
                    "title": meta["article_title"],
                    "url": meta.get("article_url", ""),
                    "date": meta.get("article_date", "")[:10],
                    "categories": meta.get("article_categories", [])
                })
        return sources


def main():
    """Test ArticleQABot with sample query."""
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY environment variable")
        return

    print("=== Initializing Article Q&A Bot ===\n")
    db = MultimodalDB(persist_directory="../data/qdrant_db")
    retriever = MultimodalRetriever(db=db)
    bot = ArticleQABot(retriever=retriever)

    # Test basic query
    query = "What are the latest developments in AI models?"
    print(f"Question: {query}\n")
    result = bot.answer_question(query, n_results=3)

    print("=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    for i, src in enumerate(result["sources"], 1):
        print(f"{i}. {src['title']} ({src['date']}) - {', '.join(src['categories'])}")
    print(f"\n=== Images: {len(result['images'])} found ===")

    # Test with filter
    print("\n\n=== Testing with Category Filter ===")
    ml_filter = build_filter(categories=["ML Research"])
    filtered = bot.answer_question(
        "What's new in machine learning?", n_results=3, where=ml_filter
    )
    print(f"Answer: {filtered['answer']}\nSources: {len(filtered['sources'])}")


if __name__ == "__main__":
    main()
