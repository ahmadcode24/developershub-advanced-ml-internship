"""
chatbot.py
----------
Context-aware chatbot with conversational memory.
"""

from typing import List, Dict


class ConversationMemory:
    """
    Simple conversation memory that stores chat history.
    """

    def __init__(self, max_history: int = 5):
        self.history = []
        self.max_history = max_history

    def add_turn(self, user_query: str, assistant_response: str):
        """Add a conversation turn to memory."""
        self.history.append({'role': 'user', 'content': user_query})
        self.history.append({'role': 'assistant', 'content': assistant_response})

        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def get_context(self) -> str:
        """Get the conversation history as a formatted string."""
        context = []
        for turn in self.history:
            prefix = 'User' if turn['role'] == 'user' else 'Assistant'
            context.append(f"{prefix}: {turn['content']}")
        return '
'.join(context)

    def get_enhanced_query(self, current_query: str) -> str:
        """Enhance the current query with conversation context."""
        if not self.history:
            return current_query

        last_user_queries = [t['content'] for t in self.history if t['role'] == 'user']
        if last_user_queries:
            context = ' '.join(last_user_queries[-2:])
            return f"{context} {current_query}"

        return current_query

    def clear(self):
        """Clear conversation history."""
        self.history = []


class RAGChatbot:
    """
    Context-aware chatbot using Retrieval-Augmented Generation.
    """

    def __init__(self, rag_pipeline, memory: ConversationMemory, top_k: int = 3):
        self.rag_pipeline = rag_pipeline
        self.memory = memory
        self.top_k = top_k

    def chat(self, query: str) -> Dict:
        """
        Process a user query and return a response.

        Parameters
        ----------
        query : str
            User query.

        Returns
        -------
        Dict
            Response with answer, sources, and scores.
        """
        enhanced_query = self.memory.get_enhanced_query(query)
        retrieved = self.rag_pipeline.retrieve(enhanced_query, k=self.top_k)
        answer = self.rag_pipeline.generate_answer(query, retrieved)
        self.memory.add_turn(query, answer)

        return {
            'query': query,
            'enhanced_query': enhanced_query,
            'answer': answer,
            'sources': [doc['doc_title'] for doc in retrieved],
            'similarity_scores': [doc['similarity_score'] for doc in retrieved],
            'retrieved_chunks': retrieved
        }

    def get_conversation_history(self) -> str:
        """Get the full conversation history."""
        return self.memory.get_context()

    def reset(self):
        """Reset the conversation."""
        self.memory.clear()
