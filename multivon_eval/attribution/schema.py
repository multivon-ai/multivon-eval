"""Dataclasses for the attribution package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PromptRecord:
    """A single prompt literal found at an SDK call site.

    Two PromptRecords are considered the "same call site across refs" iff
    (file_path, line, sdk, role, role_position) all match. This is the
    Tier-1 identity used by diff_records. A future Tier-2 step could add
    file-move / call-site-shift detection.
    """
    file_path: str        # repo-root-relative path
    line: int             # 1-indexed line of the kwarg or list element
    sdk: str              # "anthropic" | "openai" | "litellm"
    call_site: str        # short label, e.g. "messages.create" or "chat.completions.create"
    role: str             # "system" | "user" | "assistant" | "developer"
    role_position: int    # index in the messages array; -1 for the system= kwarg
    qualname: str         # fully-qualified name of the enclosing function, or "<module>"
    text: str             # the actual prompt literal (after f-string concat for pure-literal f-strings)
    is_dynamic: bool      # True iff the literal contains runtime interpolation we couldn't resolve
    fingerprint: str      # SHA-256 hex digest of normalized(text)

    @property
    def call_site_id(self) -> str:
        """A stable human-readable id like 'extractors/invoice.py:42:anthropic.system'."""
        suffix = self.role if self.role_position < 0 else f"{self.role}#{self.role_position}"
        return f"{self.file_path}:{self.line}:{self.sdk}.{suffix}"


@dataclass(frozen=True)
class PromptDiff:
    """One prompt's change across two refs.

    change_type semantics:
      - "added":     before is None, after has the new record
      - "removed":   before has the old record, after is None
      - "modified":  before and after both present with different fingerprints
      - "dynamic":   either record is is_dynamic; text cannot be reliably compared
    """
    call_site_id: str
    change_type: str
    before: Optional[PromptRecord]
    after: Optional[PromptRecord]

    @property
    def file_path(self) -> str:
        return (self.after or self.before).file_path

    @property
    def role(self) -> str:
        return (self.after or self.before).role

    @property
    def sdk(self) -> str:
        return (self.after or self.before).sdk
