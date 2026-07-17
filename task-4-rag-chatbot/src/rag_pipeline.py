"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation pipeline with FAISS and sentence transformers.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using FAISS and sentence transformers.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the RAG pipeline.

        Parameters
        ----------
        model_name : str
            Sentence transformer model name.
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunk_metadata = None
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

    def build_index(self, chunks: List[str], chunk_metadata: List[Dict]):
        """
        Build the FAISS index from text chunks.

        Parameters
        ----------
        chunks : List[str]
            List of text chunks.
        chunk_metadata : List[Dict]
            Metadata for each chunk.
        """
        self.chunk_metadata = chunk_metadata

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            chunks, show_progress_bar=True, convert_to_numpy=True
        )

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))

        print(f"Index built: {self.index.ntotal} vectors, {self.dimension} dimensions")

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query.

        Parameters
        ----------
        query : str
            User query.
        k : int
            Number of results to retrieve.

        Returns
        -------
        List[Dict]
            Retrieved chunks with metadata and scores.
        """
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                'rank': i + 1,
                'doc_title': self.chunk_metadata[idx]['doc_title'],
                'chunk_text': self.chunk_metadata[idx]['chunk_text'],
                'distance': float(dist),
                'similarity_score': 1 / (1 + float(dist))
            })

        return results

    def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate an answer from retrieved documents.

        In production, replace with LLM generation.
        """
        context = '

'.join([doc['chunk_text'] for doc in retrieved_docs])
        sentences = context.split('. ')
        best_sentence = sentences[0] if sentences else context[:200]

        answer = f"Based on the retrieved information from {len(retrieved_docs)} documents:

"
        answer += f"{best_sentence.strip()}."

        return answer
