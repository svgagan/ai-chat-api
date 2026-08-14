# app/services/conversation_service.py
from app.services.ai_service import ai_service
from app.config import ai_config

# In-memory store — single source of truth for all conversations
# Key: session_id (string)
# Value: list of message dicts [{"role": ..., "content": ...}]
_conversations: dict = {}

# Maximum number of messages to keep per session
# System prompt + last N turns
# Prevents context window overflow
MAX_MESSAGES = 20

class ConversationService:

    def get_or_create_session(self, session_id: str, system_prompt: str) -> list:
        """
        Load existing conversation or start a fresh one.
        System prompt is always the first message.
        """
        if session_id not in _conversations:
            # New session — initialize with system prompt only
            _conversations[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
        return _conversations[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """Add a single message to the conversation history."""
        if session_id in _conversations:
            _conversations[session_id].append({
                "role": role,
                "content": content
            })

    def trim_if_needed(self, session_id: str):
        """
        Sliding window — keep only the last MAX_MESSAGES messages.
        Always preserve the system prompt (index 0).
        This prevents context window overflow on long conversations.
        """
        if session_id not in _conversations:
            return

        messages = _conversations[session_id]

        if len(messages) > MAX_MESSAGES:
            # Keep system prompt (index 0) + last (MAX_MESSAGES - 1) messages
            system_prompt_message = messages[0]
            recent_messages = messages[-(MAX_MESSAGES - 1):]
            _conversations[session_id] = [system_prompt_message] + recent_messages

    def get_history(self, session_id: str) -> list:
        """Return full conversation history for a session."""
        return _conversations.get(session_id, [])

    def clear_session(self, session_id: str):
        """
        Delete a session entirely.
        Useful for: user logs out, starts fresh, privacy requirements.
        """
        if session_id in _conversations:
            del _conversations[session_id]

    def chat(self, session_id: str, user_message: str,
             system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Main method — full conversation turn.
        Load history → add user message → get AI response → save response.
        """
        # Step 1 — load or create session
        self.get_or_create_session(session_id, system_prompt)

        # Step 2 — add user's new message to history
        self.add_message(session_id, "user", user_message)

        # Step 3 — trim if history is getting too long
        self.trim_if_needed(session_id)

        # Step 4 — get full history and send to AI
        messages = self.get_history(session_id)
        ai_response = ai_service.chat_with_history(messages)

        # Step 5 — save AI response to history
        self.add_message(session_id, "assistant", ai_response)

        return ai_response


conversation_service = ConversationService()