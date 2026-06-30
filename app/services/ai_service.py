# app/services/ai_service.py
import litellm
import instructor
from app.config import ai_config

# Tell LiteLLM which API key to use
# LiteLLM reads this before every call
litellm.api_key = ai_config.API_KEY

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

# Single instance — same singleton pattern as config
ai_service = AIService()