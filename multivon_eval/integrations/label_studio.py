"""Exchange saved text/trace trials with Label Studio's native JSON format.

Label Studio owns annotation UI, users and storage. This adapter binds exported
reviews to immutable trial evidence; it does not authenticate reviewers or turn
model annotations into human ground truth.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Literal

from ..case_manifest import canonical_json, digest
from ..result import EvalReport
from ..trials import trial_integrity_issues

ReviewerKind = Literal["human", "model", "synthetic"]

LABEL_CONFIG = """<View>
  <Header value="Review instructions"/>
  <Text name="rubric" value="$rubric"/>
  <Header value="Input"/><Text name="input" value="$input"/>
  <Header value="Reference"/><Text name="reference" value="$reference"/>
  <Header value="Context"/><Text name="context" value="$context"/>
  <Header value="Observed trace"/><Text name="trace" value="$trace"/>
  <Header value="Output"/><Text name="output" value="$output"/>
  <Header value="Execution error, if recorded"/>
  <Text name="execution_error" value="$execution_error"/>
  <Choices name="verdict" toName="output" choice="single" required="true">
    <Choice value="Accept"/><Choice value="Reject"/><Choice value="Unknown"/>
  </Choices>
  <TextArea name="review_reason" toName="output" required="true"
            placeholder="Explain your decision and any missing evidence"/>
</View>"""


def _text(value) -> str:
    return value if isinstance(value, str) else canonical_json(value)


def export_review_tasks(report: EvalReport, evaluator: str, *, rubric: str) -> list[dict]:
    """Build importable tasks, one per saved trial, with grader verdicts hidden.

    Save the returned array as JSON and retain it as the import contract. Use
    LABEL_CONFIG in a Label Studio project. This template displays text and
    traces; review original media separately in Inspect or a media-capable project.
    """
    if not isinstance(evaluator, str) or not evaluator.strip():
        raise ValueError("An explicit evaluator name is required")
    if not isinstance(rubric, str) or not rubric.strip():
        raise ValueError("An explicit review rubric is required")
    tasks = []
    for row in report.case_results:
        if not row.trials or row.evidence_error or trial_integrity_issues(row):
            raise ValueError("Review export requires intact saved trials")
        for trial in row.trials:
            evidence = trial.data
            if len([e for e in evidence["evaluators"] if e["name"] == evaluator]) != 1:
                raise ValueError(f"Trial must contain exactly one {evaluator!r} result")
            case = evidence["evaluation_case"]
            binding = {"schema": "multivon.review-binding/v1", "trial_digest": trial.digest,
                       "case_id": evidence["case_id"], "case_digest": evidence["case_digest"],
                       "source_id": evidence["case"].get("source_id"), "evaluator": evaluator}
            data = {"binding": binding, "rubric": rubric, "input": case["input"],
                    "reference": _text(case.get("expected_output")),
                    "context": _text(case.get("context")), "output": evidence["output"],
                    "trace": _text(evidence.get("agent_trace")),
                    "execution_error": evidence.get("model_error") or ""}
            data["review_key"] = digest(data)
            tasks.append({"data": data})
    if not tasks:
        raise ValueError("No saved trials to review")
    if len({t["data"]["review_key"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate trial review tasks")
    return tasks


@dataclass(frozen=True)
class ReviewAnnotation:
    review_key: str
    trial_digest: str
    case_id: str
    source_id: str | None
    evaluator: str
    label: bool | None
    reviewer: str
    reviewer_kind: ReviewerKind
    annotation_id: str
    reason: str
    created_at: str | None
    cancelled: bool = False

    def __post_init__(self) -> None:
        for value in (self.review_key, self.trial_digest, self.case_id, self.evaluator,
                      self.reviewer, self.annotation_id, self.reason):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Review identifiers and explanation must be nonempty strings")
        if self.label is not None and type(self.label) is not bool:
            raise ValueError("Review label must be boolean or unknown")
        if self.reviewer_kind not in {"human", "model", "synthetic"}:
            raise ValueError("Invalid reviewer kind")
        if type(self.cancelled) is not bool or (self.cancelled and self.label is not None):
            raise ValueError("Cancelled reviews cannot provide measured labels")
        if self.created_at is not None and not isinstance(self.created_at, str):
            raise ValueError("created_at must be a string or absent")
        if self.source_id is not None and (not isinstance(self.source_id, str) or not self.source_id):
            raise ValueError("source_id must be a nonempty string or absent")

    def to_dict(self) -> dict:
        return asdict(self)


def _task_contract(tasks: list[dict]) -> dict[str, dict]:
    if not isinstance(tasks, list):
        raise ValueError("Original review tasks must be a JSON array")
    expected = {}
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
            raise ValueError("Review tasks must contain a data object")
        data = task["data"]
        key = data.get("review_key")
        binding = data.get("binding")
        if (not isinstance(binding, dict) or
                binding.get("schema") != "multivon.review-binding/v1"):
            raise ValueError("Unsupported review binding")
        if digest({k: v for k, v in data.items() if k != "review_key"}) != key:
            raise ValueError("Review task data does not match its content digest")
        if key in expected:
            raise ValueError("Duplicate review key")
        expected[key] = data
    if not expected:
        raise ValueError("A nonempty original review task array is required")
    return expected


def _reviewer(annotation: dict) -> str:
    who = annotation.get("completed_by")
    if isinstance(who, dict):
        who = who.get("id")
    if isinstance(who, bool) or not isinstance(who, (str, int)) or not str(who).strip():
        raise ValueError("A completed_by reviewer ID is required")
    return str(who)


def import_review_annotations(exported: list[dict], original_tasks: list[dict], *,
                              reviewer_kind: ReviewerKind) -> list[ReviewAnnotation]:
    """Read Label Studio JSON exports against the retained original task array.

    Predictions and drafts are not completed reviews. Cancelled annotations are
    retained as unknown. The caller must explicitly declare reviewer kind; an
    exported user ID alone cannot establish that a label is human or independent.
    """
    if reviewer_kind not in {"human", "model", "synthetic"}:
        raise ValueError("reviewer_kind must be human, model or synthetic")
    expected = _task_contract(original_tasks)
    if not isinstance(exported, list):
        raise ValueError("Use Label Studio's full JSON array export")
    records, seen_tasks, seen_annotations = [], set(), set()
    for task in exported:
        if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
            raise ValueError("Exported tasks must contain a data object")
        data = task["data"]
        key = data.get("review_key")
        if (not isinstance(key, str) or key not in expected or
                canonical_json(data) != canonical_json(expected[key])):
            raise ValueError("Exported task differs from its original review evidence")
        if key in seen_tasks:
            raise ValueError("Duplicate exported task")
        seen_tasks.add(key)
        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError("annotations must be an array")
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("Each annotation must be an object")
            annotation_id = annotation.get("id")
            if isinstance(annotation_id, bool) or not isinstance(annotation_id, (str, int)):
                raise ValueError("A stable annotation ID is required")
            annotation_id = str(annotation_id)
            if not annotation_id.strip() or annotation_id in seen_annotations:
                raise ValueError("Empty or duplicate annotation ID")
            seen_annotations.add(annotation_id)
            reviewer = _reviewer(annotation)
            cancelled = annotation.get("was_cancelled", False)
            if type(cancelled) is not bool:
                raise ValueError("was_cancelled must be boolean")
            results = annotation.get("result", [])
            if cancelled and results in (None, {}):
                results = []
            if not isinstance(results, list) or any(not isinstance(r, dict) for r in results):
                raise ValueError("Annotation result must be an array")
            verdicts = [r for r in results if r.get("from_name") == "verdict"]
            reasons = [r for r in results if r.get("from_name") == "review_reason"]
            label, reason = None, "Cancelled annotation"
            if not cancelled:
                if len(verdicts) != 1 or len(reasons) != 1:
                    raise ValueError("Completed review needs one verdict and one explanation")
                verdict, rationale = verdicts[0], reasons[0]
                if verdict.get("type") != "choices" or verdict.get("to_name") != "output":
                    raise ValueError("Verdict must be a choices annotation on output")
                if not isinstance(verdict.get("value"), dict) or not isinstance(rationale.get("value"), dict):
                    raise ValueError("Annotation values must be objects")
                choices = verdict["value"].get("choices")
                labels = {"Accept": True, "Reject": False, "Unknown": None}
                if (not isinstance(choices, list) or len(choices) != 1 or
                        not isinstance(choices[0], str) or choices[0] not in labels):
                    raise ValueError("Review must select Accept, Reject or Unknown")
                if rationale.get("type") != "textarea" or rationale.get("to_name") != "output":
                    raise ValueError("Review explanation must annotate output")
                text = rationale.get("value", {}).get("text")
                if not isinstance(text, list) or not text or any(not isinstance(t, str) for t in text):
                    raise ValueError("Review explanation must contain text")
                reason = "\n".join(text).strip()
                if not reason:
                    raise ValueError("Review explanation cannot be blank")
                label = labels[choices[0]]
                if data["execution_error"] and label is True:
                    raise ValueError("A model execution error cannot establish accepted output quality")
            binding = data["binding"]
            records.append(ReviewAnnotation(key, binding["trial_digest"], binding["case_id"],
                binding["source_id"], binding["evaluator"], label, reviewer, reviewer_kind,
                annotation_id, reason, annotation.get("created_at"), cancelled))
    return records


def reconcile_reviews(original_tasks: list[dict], reviews: list[ReviewAnnotation], *,
                      min_reviewers: int = 2, reviewer_kind: ReviewerKind = "human") -> dict:
    """Report unanimous labels, disagreement and missing review coverage.

    Unknown/cancelled labels cannot provide consensus. No majority vote silently
    creates ground truth. Reviewer IDs count distinct asserted identities, not
    independently verified people; preserve the original annotations for audit.
    """
    if type(min_reviewers) is not int or min_reviewers < 1:
        raise ValueError("min_reviewers must be a positive integer")
    if reviewer_kind not in {"human", "model", "synthetic"}:
        raise ValueError("Invalid reviewer_kind")
    expected = _task_contract(original_tasks)
    groups = defaultdict(list)
    for review in reviews:
        if review.review_key not in expected:
            raise ValueError("Review does not belong to this task set")
        binding = expected[review.review_key]["binding"]
        if (review.trial_digest, review.case_id, review.source_id, review.evaluator) != (
                binding["trial_digest"], binding["case_id"], binding["source_id"], binding["evaluator"]):
            raise ValueError("Review binding does not match the original task")
        groups[review.review_key].append(review)
    decisions = []
    for key in expected:
        selected = [r for r in groups[key] if r.reviewer_kind == reviewer_kind]
        if any(n > 1 for n in Counter(r.reviewer for r in selected).values()):
            raise ValueError("Multiple annotations from one reviewer need explicit adjudication")
        measured = [r for r in selected if r.label is not None and not r.cancelled]
        labels = {r.label for r in measured}
        status = ("disagreement" if len(labels) > 1 else "incomplete" if
                  len(measured) < min_reviewers or len(measured) != len(selected) else "consensus")
        decisions.append({"review_key": key, "status": status,
                          "label": measured[0].label if status == "consensus" else None,
                          "reviewers": len(selected), "measured_reviewers": len(measured)})
    return {"reviewer_kind": reviewer_kind, "min_reviewers": min_reviewers,
            "tasks": len(expected), "consensus_tasks": sum(d["status"] == "consensus" for d in decisions),
            "decisions": decisions,
            "identity_limit": "Reviewer kinds and IDs are assertions from the trusted export, not independently authenticated people"}
