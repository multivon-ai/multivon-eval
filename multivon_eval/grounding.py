"""A narrow W3C Web Annotation profile for content-bound verdict references."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from decimal import Decimal

from .media import MediaArtifact
from .result import EvalResult

CONTEXT = 'http://www.w3.org/ns/anno.jsonld'
MEDIA_FRAGMENTS = 'http://www.w3.org/TR/media-frags/'


def annotation(artifact: MediaArtifact, claim: str, *, region: tuple[int, int, int, int] | None = None,
               time: tuple[float, float] | None = None) -> dict:
    """Create a standard annotation; bounds are checked, semantic truth is not."""
    selector = []
    if time is not None:
        if len(time) != 2 or any(type(t) not in (int, float) or not math.isfinite(t) for t in time):
            raise ValueError('Supply finite start/end seconds')
        selector.append('t=npt:' + ','.join(format(Decimal(str(t)), 'f') for t in time))
    if region is not None:
        if len(region) != 4 or any(type(n) is not int for n in region):
            raise ValueError('Supply integer pixel x,y,width,height')
        selector.append('xywh=pixel:' + ','.join(str(n) for n in region))
    target = {'type': 'SpecificResource', 'source': artifact.id}
    if selector:
        target['selector'] = {'type': 'FragmentSelector', 'conformsTo': MEDIA_FRAGMENTS,
                              'value': '&'.join(selector)}
    result = {'@context': CONTEXT, 'type': 'Annotation', 'motivation': 'assessing',
              'body': {'type': 'TextualBody', 'value': claim, 'format': 'text/plain'}, 'target': target}
    validate_annotation(result, [artifact])
    return result


def validate_annotation(value: dict, artifacts: Iterable[MediaArtifact]) -> MediaArtifact:
    """Validate this profile without fetching JSON-LD contexts or media.

    Supports whole artifacts, integer pixel regions of orientation-1 still
    images and closed NPT time ranges for one primary audio/video stream.
    Rejects clipping, unknown duration, extra selectors and unsupported shapes.
    Video spatial regions are deliberately unsupported until display geometry
    (rotation/aspect ratio) can be validated. Use explicitly extracted frames.
    """
    registry = {}
    for artifact in artifacts:
        if artifact.id in registry and registry[artifact.id].data != artifact.data:
            raise ValueError('Conflicting descriptors for the same content')
        registry[artifact.id] = artifact
    if (not isinstance(value, dict) or set(value) != {'@context', 'type', 'motivation', 'body', 'target'}
            or value['@context'] != CONTEXT or value['type'] != 'Annotation' or value['motivation'] != 'assessing'):
        raise ValueError('Unsupported annotation profile')
    body, target = value['body'], value['target']
    if (not isinstance(body, dict) or set(body) != {'type', 'value', 'format'}
            or body['type'] != 'TextualBody' or body['format'] != 'text/plain'
            or not isinstance(body['value'], str) or not body['value'].strip()):
        raise ValueError('A grounding annotation needs a nonempty textual claim')
    if (not isinstance(target, dict) or set(target) not in ({'type', 'source'}, {'type', 'source', 'selector'})
            or target['type'] != 'SpecificResource' or not isinstance(target['source'], str)
            or target['source'] not in registry):
        raise ValueError('Grounding target is not bound to known content')
    artifact = registry[target['source']]
    if 'selector' not in target:
        return artifact
    selector = target['selector']
    if (not isinstance(selector, dict) or set(selector) != {'type', 'conformsTo', 'value'}
            or selector['type'] != 'FragmentSelector' or selector['conformsTo'] != MEDIA_FRAGMENTS
            or not isinstance(selector['value'], str)):
        raise ValueError('Unsupported fragment selector')
    props, seen = artifact.data['properties'], set()
    for part in selector['value'].split('&'):
        name, _, text = part.partition('=')
        if name in seen:
            raise ValueError('Duplicate selector dimension')
        seen.add(name)
        if name == 'xywh':
            if not artifact.media_type.startswith('image/') or props['orientation'] != 1:
                raise ValueError('Pixel regions require an orientation-1 still image')
            if not re.fullmatch(r'pixel:\d+,\d+,\d+,\d+', text):
                raise ValueError('Only explicit integer pixel rectangles are supported')
            x, y, width, height = map(int, text.removeprefix('pixel:').split(','))
            if width <= 0 or height <= 0 or x + width > props['width'] or y + height > props['height']:
                raise ValueError('Region exceeds the image; selectors are not silently clipped')
        elif name == 't':
            if not artifact.media_type.startswith(('audio/', 'video/')):
                raise ValueError('Time ranges require audio or video')
            if not re.fullmatch(r'npt:\d+(?:\.\d+)?,\d+(?:\.\d+)?', text):
                raise ValueError('Only closed nonnegative NPT seconds are supported')
            start, end = map(Decimal, text.removeprefix('npt:').split(','))
            if props['duration_seconds'] is None or not 0 <= start < end <= Decimal(str(props['duration_seconds'])):
                raise ValueError('Time range exceeds known media duration')
        else:
            raise ValueError('Unsupported fragment dimension')
    return artifact


def grounded_result(evaluator: str, passed: bool | None, reason: str, *,
                    annotations: list[dict], artifacts: Iterable[MediaArtifact]) -> EvalResult:
    """Attach checked references to a caller-supplied verdict, without inferring truth.

    Known verdicts require at least one reference. The caller owns semantic
    validation and the check's version; this function proves neither causality
    nor independent observation. Bytes must be verified separately at use time.
    """
    if passed is not None and type(passed) is not bool:
        raise ValueError('Grounded verdict must be boolean or None')
    if not isinstance(evaluator, str) or not evaluator.strip() or not isinstance(reason, str) or not reason.strip():
        raise ValueError('Supply a named check and nonempty verdict reason')
    if not isinstance(annotations, list) or (passed is not None and not annotations):
        raise ValueError('Measured verdicts require at least one annotation')
    artifacts = tuple(artifacts)
    for item in annotations:
        validate_annotation(item, artifacts)
    import json

    from .case_manifest import canonical_json
    metadata = {'grounding_profile': 'multivon.w3c-grounding/v1',
                'annotations': json.loads(canonical_json(annotations)),
                'artifacts': [a.data for a in artifacts],
                'verification': 'reference structure/bounds only; verdict truth is caller-supplied'}
    if passed is None:
        metadata['skipped'] = True
    return EvalResult(evaluator, float(passed is True), passed is True, reason, metadata)
