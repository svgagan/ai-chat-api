# app/services/ai_service.py
import litellm
import os
import instructor
from app.config import ai_config
from typing import Generator

def _set_api_key():
    """
    Model-agnostic API key setup for chat model.
    Detects provider from model string, sets correct env variable.
    Add new provider here when needed — nowhere else changes.
    """
    model = ai_config.MODEL
    key = ai_config.API_KEY

    if not key:
        return  # no key provided

    if model.startswith("gemini/") or model.startswith("google/"):
        os.environ["GEMINI_API_KEY"] = key
        os.environ["GOOGLE_API_KEY"] = key
    elif model.startswith("openai/"):
        os.environ["OPENAI_API_KEY"] = key
    elif model.startswith("groq/"):
        os.environ["GROQ_API_KEY"] = key
    elif model.startswith("anthropic/"):
        os.environ["ANTHROPIC_API_KEY"] = key
    elif model.startswith("ollama/"):
        pass  # local, no key needed

# Run once at startup
_set_api_key()

class AIService:
    """
    All AI interactions go through this class.
    Nothing outside this class touches LiteLLM directly.
    """

    def __init__(self):
        self.model = ai_config.MODEL
        self.default_temperature = ai_config.DEFAULT_TEMPERATURE
        self.default_max_tokens = ai_config.DEFAULT_MAX_TOKENS

        # instructor wraps litellm and adds schema enforcement + auto-retry
        self.structured_client = instructor.from_litellm(litellm.completion)

    def chat(
        self,
        user_message: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Send a message to the AI and get a response.

        Args:
            user_message:  What the user typed
            system_prompt: Instructions that shape how the AI behaves
            temperature:   Override default randomness if needed
            max_tokens:    Override default response length if needed

        Returns:
            AI response as a plain string
        """

        # Build the messages list — this is the universal format
        # Every LLM provider understands this structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]

        # One LiteLLM call works for every provider
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature or self.default_temperature,
            max_tokens=max_tokens or self.default_max_tokens
        )

        # Extract the text from the response object
        return response.choices[0].message.content

    def extract_structured(self, user_message: str, response_model, system_prompt: str = None):
        """
        Extract structured data from unstructured text.
        response_model: the Pydantic schema we want the AI to fill in.
        instructor handles: schema enforcement, validation, and automatic
        retry-with-feedback if the AI's first response fails validation.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        result = self.structured_client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=response_model,   # <-- this is the key difference
            max_retries=2                    # <-- self-healing retry, capped
        )
        return result

    def stream_chat(self, user_message: str,
                    system_prompt: str = "You are a helpful assistant.") -> Generator:
        """
        Stream AI response token by token.
        Returns a generator — caller gets one token at a time
        instead of waiting for the complete response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]

        # stream=True tells LiteLLM to return tokens as they are generated
        # instead of waiting for the complete response
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.default_temperature,
            max_tokens=self.default_max_tokens,
            stream=True         # ← this is the only difference from chat()
        )

        # response is now an iterator, not a complete object
        # each chunk contains one or more tokens as they arrive
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is not None:
                yield token     # send this token immediately, do not wait

    def chat_with_history(self, messages: list) -> str:
        """
        Send full conversation history to AI and get response.
        Unlike chat(), this accepts a pre-built messages list
        instead of constructing it internally.
        The caller is responsible for building the correct message history.
        """
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.default_temperature,
            max_tokens=self.default_max_tokens
        )
        return response.choices[0].message.content

    def chat_with_tools(
        self,
        messages: list,
        tools: list,
        tool_choice: str = "auto"
    ):
        """
        Send messages with available tools. Returns the raw response
        object — caller must check response.choices[0].message
        for whether the model wants to call a tool, or has given
        a direct text answer.
        """
        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        return response

# Single instance — same singleton pattern as config
ai_service = AIService()