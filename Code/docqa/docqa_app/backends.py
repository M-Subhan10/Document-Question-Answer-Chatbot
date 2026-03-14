from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Iterable

from .config import Settings


class BackendError(RuntimeError):
    pass


class BaseBackend:
    name = "backend"

    def ocr_image(self, image, prompt: str) -> str:
        raise NotImplementedError

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def stream(self, prompt: str) -> Iterable[str]:
        yield self.complete(prompt)


@dataclass
class OpenAIBackend(BaseBackend):
    settings: Settings
    name: str = "openai"

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=self.settings.openai_api_key)

    @staticmethod
    def _img_to_b64(image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def ocr_image(self, image, prompt: str) -> str:
        b64 = self._img_to_b64(image)
        response = self.client.chat.completions.create(
            model=self.settings.openai_ocr_model,
            max_tokens=4096,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                        },
                    ],
                }
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            max_tokens=4096,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    def stream(self, prompt: str) -> Iterable[str]:
        stream = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            max_tokens=4096,
            temperature=0.2,
            stream=True,
            messages=[{"role": "user", "content": prompt}],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


@dataclass
class GeminiBackend(BaseBackend):
    settings: Settings
    name: str = "gemini"

    def __post_init__(self) -> None:
        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        self.genai = genai
        self.ocr_model = genai.GenerativeModel(self.settings.gemini_ocr_model)
        self.chat_model = genai.GenerativeModel(self.settings.gemini_chat_model)

    def ocr_image(self, image, prompt: str) -> str:
        response = self.ocr_model.generate_content(
            [prompt, image],
            generation_config=self.genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )
        return response.text.strip()

    def complete(self, prompt: str) -> str:
        response = self.chat_model.generate_content(
            prompt,
            generation_config=self.genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        return response.text.strip()

    def stream(self, prompt: str) -> Iterable[str]:
        try:
            response = self.chat_model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception:
            yield self.complete(prompt)


def select_backends(settings: Settings) -> tuple[BaseBackend, BaseBackend | None]:
    available: dict[str, BaseBackend] = {}
    if settings.has_openai:
        available["openai"] = OpenAIBackend(settings)
    if settings.has_gemini:
        available["gemini"] = GeminiBackend(settings)
    if not available:
        raise BackendError(
            "No API key found. Set OPENAI_API_KEY or GEMINI_API_KEY before launching."
        )

    if settings.forced_backend:
        if settings.forced_backend not in available:
            raise BackendError(f"Forced backend '{settings.forced_backend}' is not available.")
        primary = available[settings.forced_backend]
    elif "openai" in available:
        primary = available["openai"]
    else:
        primary = next(iter(available.values()))

    backup = None
    for name, backend in available.items():
        if name != primary.name:
            backup = backend
            break
    return primary, backup
