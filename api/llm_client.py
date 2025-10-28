# --------------------------------------------
# LLM Client - Unified interface for Ollama, Deepseek, OpenAI
# --------------------------------------------

import json
import requests
from typing import Dict, Any, Optional
import litellm
from litellm import completion
from config import (
    DEFAULT_LLM_PROVIDER,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_BASE,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_TIMEOUT
)


class LLMClient:
    """
    Unified LLM client that works with multiple providers.
    Automatically switches based on configuration.
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM client.

        Args:
            provider: "ollama", "deepseek", "openai", or "anthropic". If None, uses DEFAULT_LLM_PROVIDER
        """
        self.provider = provider or DEFAULT_LLM_PROVIDER
        self.available = False
        self.model = None
        self.api_base = None

        self._configure()

    def _configure(self):
        """Configure the LLM client based on provider."""
        if self.provider == "ollama":
            self._configure_ollama()
        elif self.provider == "deepseek":
            self._configure_deepseek()
        elif self.provider == "openai":
            self._configure_openai()
        elif self.provider == "anthropic":
            self._configure_anthropic()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _configure_ollama(self):
        """Configure Ollama (local deployment)."""
        self.model = OLLAMA_MODEL
        self.api_base = OLLAMA_API_BASE

        try:
            # Health check
            r = requests.get(f"{self.api_base}/api/tags", timeout=10)
            r.raise_for_status()
            models = [m.get("name") for m in (r.json().get("models") or [])]

            if self.model not in models:
                self.available = False
                self.error = f"Model '{self.model}' not found. Available: {models}"
            else:
                self.available = True
        except Exception as e:
            self.available = False
            self.error = f"Ollama not reachable at {self.api_base}: {e}"

    def _configure_deepseek(self):
        """Configure Deepseek (cloud deployment)."""
        if not DEEPSEEK_API_KEY:
            self.available = False
            self.error = "DEEPSEEK_API_KEY not configured"
            return

        self.model = DEEPSEEK_MODEL
        self.api_base = DEEPSEEK_API_BASE
        self.available = True

    def _configure_openai(self):
        """Configure OpenAI (cloud deployment)."""
        if not OPENAI_API_KEY:
            self.available = False
            self.error = "OPENAI_API_KEY not configured"
            return

        self.model = OPENAI_MODEL
        self.api_base = "https://api.openai.com/v1"
        self.available = True

    def _configure_anthropic(self):
        """Configure Anthropic/Claude (cloud deployment)."""
        if not ANTHROPIC_API_KEY:
            self.available = False
            self.error = "ANTHROPIC_API_KEY not configured"
            return

        self.model = ANTHROPIC_MODEL
        self.api_base = "https://api.anthropic.com"
        self.available = True

    def get_status(self) -> Dict[str, Any]:
        """
        Get status information about the LLM client.

        Returns:
            Dict with status, provider, model, and error info
        """
        return {
            "available": self.available,
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "error": getattr(self, 'error', None)
        }

    def complete(self, prompt: str, temperature: float = 0.7,
                max_tokens: int = 2000, system_prompt: Optional[str] = None) -> str:
        """
        Generate completion using configured LLM provider.

        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            Generated text or error message
        """
        if not self.available:
            return f"Error: LLM not available. {getattr(self, 'error', '')}"

        try:
            if self.provider == "ollama":
                return self._complete_ollama(prompt, temperature, max_tokens)
            else:
                return self._complete_litellm(prompt, temperature, max_tokens, system_prompt)
        except Exception as e:
            return f"Error during LLM completion: {e}"

    def _complete_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Direct Ollama API call."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            headers = {"Content-Type": "application/json"}
            r = requests.post(
                f"{self.api_base}/api/generate",
                headers=headers,
                data=json.dumps(payload),
                timeout=LLM_TIMEOUT
            )
            r.raise_for_status()
            result = r.json()
            return result.get("response", "No response received.")
        except requests.exceptions.Timeout:
            return "Error: LLM request timed out."
        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to Ollama at {self.api_base}"
        except Exception as e:
            return f"Error: {e}"

    def _complete_litellm(self, prompt: str, temperature: float,
                         max_tokens: int, system_prompt: Optional[str] = None) -> str:
        """Use litellm for API-based providers (Deepseek, OpenAI)."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Set API key based on provider
            if self.provider == "deepseek":
                litellm.api_key = DEEPSEEK_API_KEY
                model_name = f"deepseek/{self.model}"
            elif self.provider == "openai":
                litellm.api_key = OPENAI_API_KEY
                model_name = self.model
            elif self.provider == "anthropic":
                litellm.api_key = ANTHROPIC_API_KEY
                model_name = self.model
            else:
                model_name = self.model

            response = completion(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=LLM_TIMEOUT,
                api_base=self.api_base if self.provider == "deepseek" else None
            )

            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
                # Clean up response
                lines = [ln.rstrip() for ln in content.splitlines()]
                return "\n".join(lines)

            return "No valid response from LLM."
        except litellm.exceptions.Timeout:
            return "Error: LLM request timed out."
        except litellm.exceptions.AuthenticationError:
            return f"Error: Authentication failed for {self.provider}. Check API key."
        except Exception as e:
            return f"Error: {e}"


# Create singleton instance
_llm_client = None


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """
    Get or create LLM client singleton.

    Args:
        provider: Optional provider override

    Returns:
        LLMClient instance
    """
    global _llm_client
    if _llm_client is None or provider:
        _llm_client = LLMClient(provider)
    return _llm_client
