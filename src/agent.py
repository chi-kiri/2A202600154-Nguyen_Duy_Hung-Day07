from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        results = self._store.search(question, top_k=top_k)

        if not results:
            return "I don't know."

        context_parts = []
        for i, r in enumerate(results):
            content = r.get("content", "")
            context_parts.append(f"[Chunk {i + 1}]\n{content}")

        context = "\n\n".join(context_parts)

        prompt = f"""
        You are a helpful assistant. Answer the question based ONLY on the context below.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        answer = self._llm_fn(prompt)

        return answer.strip()
        # raise NotImplementedError("Implement KnowledgeBaseAgent.answer")
