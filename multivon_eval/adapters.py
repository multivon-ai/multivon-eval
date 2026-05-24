"""
Model adapters — callable wrappers around LLM clients.

Subclass ModelAdapter to add custom behavior: retry logic, prompt templating,
structured output parsing, cost tracking, etc. The adapter is passed directly
to suite.run() since it implements __call__(input: str) -> str.

    class MyAdapter(ModelAdapter):
        def __call__(self, input: str) -> str:
            return my_client.generate(self._build_prompt(input))

    report = suite.run(MyAdapter())

Built-in adapters (``OpenAIAdapter``, ``AnthropicAdapter``, ``LiteLLMAdapter``)
also implement ``_call_with_case(case)`` so they receive the full ``EvalCase``
when the suite runs them, automatically injecting any ``case.context`` into
the system prompt for RAG cases. Custom adapters that only override
``__call__`` get the string-in/string-out behavior unchanged.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .case import EvalCase


def _format_context_block(context: Any) -> str:
    """Turn ``case.context`` (str | list[str] | None) into a system-prompt block.

    Returns the empty string if there's no context. RAG users typically want
    the chunks separated visually so the model knows where one ends and the
    next begins.
    """
    if context is None or context == "":
        return ""
    if isinstance(context, str):
        return f"\n\nContext:\n{context}\n"
    if isinstance(context, list):
        chunks = "\n\n".join(
            f"[chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context) if chunk
        )
        return f"\n\nContext:\n{chunks}\n"
    return ""


_RAG_SYSTEM_PREFIX = (
    "You are answering based on the provided context. Use ONLY the information "
    "in the context below to answer the question. If the answer is not in the "
    "context, say so explicitly rather than guessing."
)


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

    def _call_with_case(self, case: "EvalCase") -> str:
        """Context-aware entry point used by ``suite.run()`` when available.

        Auto-injects ``case.context`` into the system prompt so RAG cases work
        with ``run_with_openai`` out of the box.
        """
        client = self._get_client()
        ctx_block = _format_context_block(case.context)
        messages: list[dict[str, str]] = []
        if ctx_block:
            base_system = (self._system_prompt + "\n\n" if self._system_prompt else "") \
                + _RAG_SYSTEM_PREFIX + ctx_block
            messages.append({"role": "system", "content": base_system})
        elif self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": case.input})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._extra,
        )
        return response.choices[0].message.content or ""


class LiteLLMAdapter(ModelAdapter):
    """
    Adapter for LiteLLM — covers 100+ providers with one interface.

    Use this instead of writing a custom adapter for Azure OpenAI, AWS Bedrock,
    Google Vertex AI, Ollama, Groq, or any other provider LiteLLM supports.

    Install: pip install 'multivon-eval[litellm]'

    Args:
        model:          LiteLLM model string. Format varies by provider:
                        "gpt-4o"                               → OpenAI
                        "claude-opus-4-7"                      → Anthropic
                        "azure/gpt-4o"                         → Azure OpenAI
                        "bedrock/anthropic.claude-3-sonnet-…"  → AWS Bedrock
                        "vertex_ai/gemini-1.5-pro"             → Google Vertex
                        "ollama/llama3.2"                      → Local Ollama
                        "groq/llama-3.1-70b-versatile"         → Groq
        system_prompt:  Optional system message prepended to every call.
        temperature:    Sampling temperature (default 0.0 for determinism).
        max_tokens:     Max output tokens (default 1024).
        **kwargs:       Provider-specific kwargs forwarded to litellm.completion():
                        api_base, api_key, api_version, etc.

    Examples:

        # Azure OpenAI
        LiteLLMAdapter(
            "azure/gpt-4o",
            api_base="https://my-deployment.openai.azure.com",
            api_key=os.environ["AZURE_API_KEY"],
            api_version="2024-02-01",
        )

        # AWS Bedrock (uses boto3 credentials from env/~/.aws)
        LiteLLMAdapter("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")

        # Local Ollama
        LiteLLMAdapter("ollama/llama3.2", api_base="http://localhost:11434")

    Subclass to customise prompt construction:

        class InstructAdapter(LiteLLMAdapter):
            def _build_messages(self, input: str) -> list[dict]:
                return [{"role": "user", "content": f"### Instruction\\n{input}\\n### Response"}]
    """

    def __init__(
        self,
        model: str,
        *,
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

    def _build_messages(self, input: str) -> list[dict[str, str]]:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": input})
        return messages

    def __call__(self, input: str) -> str:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for LiteLLMAdapter: "
                "pip install 'multivon-eval[litellm]'"
            )
        response = litellm.completion(
            model=self.model,
            messages=self._build_messages(input),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._extra,
        )
        return response.choices[0].message.content or ""

    def _call_with_case(self, case: "EvalCase") -> str:
        """Context-aware entry point — auto-injects ``case.context`` for RAG."""
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for LiteLLMAdapter: "
                "pip install 'multivon-eval[litellm]'"
            )
        ctx_block = _format_context_block(case.context)
        messages: list[dict[str, str]] = []
        if ctx_block:
            base_system = (self._system_prompt + "\n\n" if self._system_prompt else "") \
                + _RAG_SYSTEM_PREFIX + ctx_block
            messages.append({"role": "system", "content": base_system})
        elif self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": case.input})

        response = litellm.completion(
            model=self.model,
            messages=messages,
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
                        Note: Anthropic's reasoning-tier models — claude-opus-4-7
                        and the claude-opus-5+ family — deprecated this
                        parameter and reject the request with a 400 if it's
                        present. We detect those models by name and silently
                        drop the field; older models still honour it. See
                        ``_supports_temperature`` below.
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

    def _supports_temperature(self) -> bool:
        """Whether this model accepts the ``temperature`` parameter.

        Anthropic's reasoning-tier models (claude-opus-4-7 and the
        claude-opus-5+ family) reject ``temperature`` with a 400. We omit
        the field for those models and let the SDK use the model's default
        (effectively deterministic for the reasoning tier). Older models
        (claude-3, claude-3-5, claude-haiku-4, claude-sonnet-4, claude-opus-3)
        still accept and honour temperature, so we keep it for them.

        Detection is name-prefix-based — covers ``claude-opus-4-7``,
        ``claude-opus-4-7-20251101``, ``claude-opus-5``, ``claude-opus-5-1``,
        etc. without listing every dated variant.
        """
        m = (self.model or "").lower()
        return not (m.startswith("claude-opus-4-7") or m.startswith("claude-opus-5"))

    def _build_messages(self, input: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": input}]

    def _build_kwargs(self, messages: list[dict], system: str = "") -> dict[str, Any]:
        """Compose the kwargs for client.messages.create.

        Centralised so both ``__call__`` and ``_call_with_case`` share the
        same logic for omitting ``temperature`` on reasoning-tier models.
        """
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens,
            **self._extra,
        )
        if self._supports_temperature():
            kwargs["temperature"] = self._temperature
        if system:
            kwargs["system"] = system
        return kwargs

    def __call__(self, input: str) -> str:
        client = self._get_client()
        kwargs = self._build_kwargs(self._build_messages(input), self._system_prompt)
        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _call_with_case(self, case: "EvalCase") -> str:
        """Context-aware entry point used by ``suite.run()`` when available.

        Auto-injects ``case.context`` into the system prompt so RAG cases work
        with ``run_with_anthropic`` out of the box. Subclasses that override
        ``__call__`` for custom prompt building can also override this method
        if they want fine-grained case access.
        """
        client = self._get_client()
        ctx_block = _format_context_block(case.context)
        if ctx_block:
            system = (self._system_prompt + "\n\n" if self._system_prompt else "") \
                + _RAG_SYSTEM_PREFIX + ctx_block
        else:
            system = self._system_prompt

        kwargs = self._build_kwargs(self._build_messages(case.input), system)
        response = client.messages.create(**kwargs)
        return response.content[0].text
