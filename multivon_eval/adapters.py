"""
Model adapters — callable wrappers around LLM clients.

Subclass ModelAdapter to add custom behavior: retry logic, prompt templating,
structured output parsing, cost tracking, etc. The adapter is passed directly
to suite.run() since it implements __call__(input: str) -> str.

    class MyAdapter(ModelAdapter):
        def __call__(self, input: str) -> str:
            return my_client.generate(self._build_prompt(input))

    report = suite.run(MyAdapter())
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    """
    Base class for model adapters.

    An adapter is a callable that takes a string input and returns a string
    output. Subclass this to wrap any model client, add a system prompt,
    inject retry logic, or post-process outputs.

    The adapter plugs directly into suite.run():

        report = suite.run(MyAdapter())

    Built-in adapters (OpenAIAdapter, AnthropicAdapter) follow the same
    interface and can also be subclassed.
    """

    @abstractmethod
    def __call__(self, input: str) -> str: ...

    def with_system_prompt(self, prompt: str) -> "_WithSystemPrompt":
        """Return a new adapter that prepends a system prompt to every call."""
        return _WithSystemPrompt(self, prompt)


class _WithSystemPrompt(ModelAdapter):
    """Wraps any ModelAdapter to inject a system prompt."""

    def __init__(self, adapter: ModelAdapter, system_prompt: str) -> None:
        self._adapter = adapter
        self._system_prompt = system_prompt

    def __call__(self, input: str) -> str:
        return self._adapter(input)

    # Override in subclasses that use the system prompt natively.
    # This wrapper exists so plain subclasses can use with_system_prompt()
    # without having to implement the plumbing themselves.


class OpenAIAdapter(ModelAdapter):
    """
    Adapter for the OpenAI Python client (openai>=1.0).

    Args:
        model:          Model ID, e.g. "gpt-4o", "gpt-4o-mini".
        client:         An openai.OpenAI instance. If None, one is created
                        from the OPENAI_API_KEY environment variable.
        system_prompt:  Optional system message prepended to every call.
        temperature:    Sampling temperature (default 0.0 for determinism).
        max_tokens:     Max output tokens (default 1024).
        **kwargs:       Extra keyword args forwarded to client.chat.completions.create().

    Subclass to customise:

        class MyOpenAIAdapter(OpenAIAdapter):
            def _build_messages(self, input: str) -> list[dict]:
                msgs = super()._build_messages(input)
                msgs[0]["content"] += "\\nAlways answer in bullet points."
                return msgs
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        client: Any = None,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = kwargs
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIAdapter: pip install openai"
            )
        return openai.OpenAI()

    def _build_messages(self, input: str) -> list[dict[str, str]]:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": input})
        return messages

    def __call__(self, input: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(input),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._extra,
        )
        return response.choices[0].message.content or ""


class AnthropicAdapter(ModelAdapter):
    """
    Adapter for the Anthropic Python client (anthropic>=0.20).

    Args:
        model:          Model ID, e.g. "claude-haiku-4-5-20251001".
        client:         An anthropic.Anthropic instance. If None, one is created
                        from the ANTHROPIC_API_KEY environment variable.
        system_prompt:  Optional system message.
        temperature:    Sampling temperature (default 0.0 for determinism).
        max_tokens:     Max output tokens (default 1024).
        **kwargs:       Extra keyword args forwarded to client.messages.create().

    Subclass to customise:

        class MyAnthropicAdapter(AnthropicAdapter):
            def _build_messages(self, input: str) -> list[dict]:
                return [{"role": "user", "content": f"Task: {input}"}]
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        client: Any = None,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra = kwargs
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter: pip install anthropic"
            )
        return anthropic.Anthropic()

    def _build_messages(self, input: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": input}]

    def __call__(self, input: str) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=self._build_messages(input),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._extra,
        )
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        response = client.messages.create(**kwargs)
        return response.content[0].text
