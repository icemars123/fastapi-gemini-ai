import os
import google.generativeai as genai
from .base import AIPlatform


class Gemini(AIPlatform):
    def __init__(self, api_key: str, system_prompt: str = None):
        self.api_key = api_key
        self.system_prompt = system_prompt
        genai.configure(api_key=self.api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(self.model_name)

    def _fallback_model_name(self) -> str | None:
        """Pick the first model that supports generateContent."""
        try:
            for model in genai.list_models():
                if "generateContent" in getattr(model, "supported_generation_methods", []):
                    return model.name.replace("models/", "")
        except Exception:
            return None
        return None

    def chat(self, prompt: str) -> str:
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"

        try:
            response = self.model.generate_content(prompt)
        except Exception as exc:
            message = str(exc)
            if "is not found" not in message and "not supported for generateContent" not in message:
                raise

            fallback = self._fallback_model_name()
            if not fallback:
                raise

            self.model_name = fallback
            self.model = genai.GenerativeModel(self.model_name)
            response = self.model.generate_content(prompt)

        return response.text
