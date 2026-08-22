# app/services/rag_service.py
from sqlalchemy.orm import Session
from app.services.embedding_service import embedding_service
from app.services.conversation_service import conversation_service
from app.services.ai_service import ai_service
from app.config import ai_config

class RAGService:

    def __init__(self):
        self.embedding_service = embedding_service
        self.ai_service = ai_service
        self.similarity_threshold = ai_config.RAG_SIMILARITY_THRESHOLD
        self.max_chunks = ai_config.RAG_MAX_CHUNKS

    def _filter_by_threshold(self, chunks: list[dict]) -> list[dict]:
        """
        Keep only chunks that meet the minimum similarity threshold.
        Prevents handing irrelevant context to the AI.
        """
        return [
            chunk for chunk in chunks
            if chunk["similarity_score"] >= self.similarity_threshold
        ]

    def _build_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a single context block
        for the prompt. Each chunk is labeled with its source
        so the AI (and eventually the user) knows where it came from.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['source']}]\n{chunk['content']}"
            )
        return "\n\n".join(context_parts)

    def answer_question(
        self,
        query: str,
        source_filter: str = None,
        db: Session = None
    ) -> dict:
        """
        Full RAG pipeline: retrieve, filter, augment, generate.
        Returns the answer plus which sources were actually used —
        so the caller can show citations.
        """
        # Step 1 — RETRIEVE
        retrieved_chunks = self.embedding_service.search(
            query=query,
            limit=self.max_chunks,
            source_filter=source_filter,
            db=db
        )

        # Step 2 — FILTER by similarity threshold
        relevant_chunks = self._filter_by_threshold(retrieved_chunks)

        # Step 3 — Handle the "nothing relevant found" case
        if not relevant_chunks:
            return {
                "answer": "I don't have information about that in the available documents.",
                "sources_used": [],
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_used": 0,
                "chunk_citations": []
            }

        # Step 4 — AUGMENT — build context and system prompt
        context = self._build_context(relevant_chunks)

        system_prompt = f"""You are a helpful assistant that answers questions
        using ONLY the context provided below. Do not use any outside knowledge.
        If the context does not contain enough information to answer the question,
        say so clearly instead of guessing.

        Context:
        {context}
        """

        # Step 5 — GENERATE
        answer = self.ai_service.chat(
            user_message=query,
            system_prompt=system_prompt
        )

        # Step 6 — return answer with source attribution
        sources_used = list(set(chunk["source"] for chunk in relevant_chunks))

        chunk_citations = [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "similarity_score": chunk["similarity_score"]
            }
            for chunk in relevant_chunks
        ]

        return {
            "answer": answer,
            "sources_used": sources_used,
            "chunks_retrieved": len(retrieved_chunks),
            "chunks_used": len(relevant_chunks),
            "chunk_citations": chunk_citations
        }

    def _rewrite_query(self, query: str, history: list) -> str:
        """
        Rewrite a follow-up query into a standalone question
        using conversation history, so retrieval can work
        without needing the prior turn's context.
        Skipped if history has no prior turns.
        """
        # history includes system prompt at index 0 — need at least
        # one prior user+assistant exchange to rewrite meaningfully
        if len(history) <= 1:
            return query

        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in history
            if msg["role"] != "system"
        )

        rewrite_prompt = f"""Given this conversation history:
            {history_text}

            Rewrite the following follow-up question into a standalone
            question that makes sense without the conversation history.
            Only output the rewritten question, nothing else.

            Follow-up question: {query}"""

        rewritten = self.ai_service.chat(
            user_message=rewrite_prompt,
            system_prompt="You rewrite follow-up questions into standalone questions.",
            temperature=0.0
        )
        return rewritten.strip()


    def answer_question_conversational(
        self,
        session_id: str,
        query: str,
        source_filter: str = None,
        system_prompt: str = "You are a helpful assistant answering questions from documents.",
        db: Session = None
    ) -> dict:
        """
        RAG with conversation memory. Follow-up questions get
        rewritten into standalone questions before retrieval.
        """
        # Step 1 — load or create session, get existing history
        history = conversation_service.get_or_create_session(
            session_id, system_prompt
        )

        # Step 2 — rewrite query if this is a follow-up
        search_query = self._rewrite_query(query, history)

        # Step 3 — retrieve using the rewritten query
        retrieved_chunks = self.embedding_service.search(
            query=search_query,
            limit=self.max_chunks,
            source_filter=source_filter,
            db=db
        )
        relevant_chunks = self._filter_by_threshold(retrieved_chunks)

        if not relevant_chunks:
            answer = "I don't have information about that in the available documents."
        else:
            context = self._build_context(relevant_chunks)
            rag_system_prompt = f"""Answer using ONLY the context below.
            If the context is insufficient, say so.

            Context:
            {context}
            """
            answer = self.ai_service.chat(
                user_message=query,
                system_prompt=rag_system_prompt
            )

        # Step 4 — save this turn to conversation history
        conversation_service.add_message(session_id, "user", query)
        conversation_service.add_message(session_id, "assistant", answer)
        conversation_service.trim_if_needed(session_id)

        sources_used = list(set(c["source"] for c in relevant_chunks)) if relevant_chunks else []

        chunk_citations = [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "similarity_score": chunk["similarity_score"]
            }
            for chunk in relevant_chunks
        ]

        return {
            "answer": answer,
            "rewritten_query": search_query,
            "sources_used": sources_used,
            "chunks_retrieved": len(retrieved_chunks),
            "chunks_used": len(relevant_chunks) if relevant_chunks else 0,
            "chunk_citations": chunk_citations
        }

rag_service = RAGService()