from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """
    A single test case for evaluation.

    Attributes:
        input:           The prompt or question sent to the model.
        expected_output: The ideal response (used by deterministic evaluators).
        context:         Source documents or retrieved chunks (used by faithfulness/hallucination evals).
        metadata:        Arbitrary key-value data attached to this case (e.g. category, source_id).
        tags:            Labels for filtering results in reports.
    """
    input: str
    expected_output: str | None = None
    context: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def context_str(self) -> str:
        if self.context is None:
            return ""
        if isinstance(self.context, list):
            return "\n\n".join(self.context)
        return self.context
