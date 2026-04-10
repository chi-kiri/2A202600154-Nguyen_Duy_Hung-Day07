from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        chunks = []
        current_chunk = []

        for sentence in sentences:
            if sentence:
                current_chunk.append(sentence.strip())

            if len(current_chunk) == self.max_sentences_per_chunk:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return chunks
        # raise NotImplementedError("Implement SentenceChunker.chunk")


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        if not text:
            return []
        return self._split(text, self.separators)
        # raise NotImplementedError("Implement RecursiveChunker.chunk")

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        if len(current_text) <= self.chunk_size:
            return [current_text.strip()]

        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size].strip()
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]

        if separator == "":
            return [
                current_text[i : i + self.chunk_size].strip()
                for i in range(0, len(current_text), self.chunk_size)
            ]

        parts = re.split(f'({re.escape(separator)})', current_text)

        chunks = []
        buffer = ""

        for part in parts:
            if buffer:
                candidate = buffer + separator + part
            else:
                candidate = part

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.extend(self._split(buffer, remaining_separators[1:]))
                buffer = part

        if buffer:
            chunks.extend(self._split(buffer, remaining_separators[1:]))

        return [c.strip() for c in chunks if c.strip()]
        # raise NotImplementedError("Implement RecursiveChunker._split")


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    if not vec_a or not vec_b:
        return 0.0

    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
    # raise NotImplementedError("Implement compute_similarity")


class SemanticChunker:
    """
    Split text into chunks based on semantic similarity between consecutive sentences.
    If the cosine similarity between two adjacent sentences drops below a threshold,
    we create a new chunk.
    """

    def __init__(self, embedding_fn, threshold: float = 0.45) -> None:
        self.embedding_fn = embedding_fn
        self.threshold = threshold

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return []
            
        embeddings = [self.embedding_fn(s) for s in sentences]
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = compute_similarity(embeddings[i-1], embeddings[i])
            
            if sim >= self.threshold:
                # Still the same topic
                current_chunk.append(sentences[i])
            else:
                # Topic changed
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        fixed = FixedSizeChunker(chunk_size=chunk_size)
        sentence = SentenceChunker()
        recursive = RecursiveChunker(chunk_size=chunk_size)

        strategies = {
            "fixed_size": fixed.chunk(text),
            "by_sentences": sentence.chunk(text),
            "recursive": recursive.chunk(text),
        }

        result = {}

        for name, chunks in strategies.items():
            lengths = [len(c) for c in chunks] if chunks else []

            result[name] = {
                "count": len(chunks),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "chunks": chunks,
            }

        return result
        # raise NotImplementedError("Implement ChunkingStrategyComparator.compare")
