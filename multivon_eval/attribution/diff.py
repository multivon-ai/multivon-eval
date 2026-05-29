"""Compute PromptDiff lists across two PromptRecord collections (base vs head)."""
from __future__ import annotations

from .schema import PromptDiff, PromptRecord


def _index_records(records: list[PromptRecord]) -> dict[str, PromptRecord]:
    """Index by call_site_id. Later occurrences (rare) overwrite earlier."""
    return {r.call_site_id: r for r in records}


def diff_records(
    base: list[PromptRecord], head: list[PromptRecord]
) -> list[PromptDiff]:
    """Compute the structured diff between two prompt extractions.

    Identity is call_site_id (file_path + line + sdk + role + role_position).
    Returns diffs in a stable order: modified first, then added, then removed,
    sorted by call_site_id within each group.

    A record on either side with is_dynamic=True produces a 'dynamic' change_type
    when present in both refs — the text isn't reliably comparable.
    """
    base_idx = _index_records(base)
    head_idx = _index_records(head)
    diffs: list[PromptDiff] = []

    common_ids = base_idx.keys() & head_idx.keys()
    for cid in common_ids:
        b, h = base_idx[cid], head_idx[cid]
        if b.is_dynamic or h.is_dynamic:
            if b.text != h.text:
                diffs.append(PromptDiff(call_site_id=cid, change_type="dynamic",
                                        before=b, after=h))
            # else: both unchanged dynamic placeholders → skip
            continue
        if b.fingerprint != h.fingerprint:
            diffs.append(PromptDiff(call_site_id=cid, change_type="modified",
                                    before=b, after=h))

    for cid in head_idx.keys() - base_idx.keys():
        diffs.append(PromptDiff(call_site_id=cid, change_type="added",
                                before=None, after=head_idx[cid]))
    for cid in base_idx.keys() - head_idx.keys():
        diffs.append(PromptDiff(call_site_id=cid, change_type="removed",
                                before=base_idx[cid], after=None))

    order = {"modified": 0, "added": 1, "removed": 2, "dynamic": 3}
    diffs.sort(key=lambda d: (order.get(d.change_type, 99), d.call_site_id))
    return diffs
