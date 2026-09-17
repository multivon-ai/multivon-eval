"""Bind explicit caller-owned bytes to native Inspect media content."""
from __future__ import annotations

import base64
from collections.abc import Callable

from ..case import EvalCase
from ..media import MediaArtifact, case_media


def to_inspect_content(artifact: MediaArtifact, content: bytes):
    from inspect_ai.model import ContentAudio, ContentDocument, ContentImage, ContentVideo
    uri = artifact.data_uri(content)
    if artifact.media_type.startswith('image/'):
        return ContentImage(image=uri)
    if artifact.media_type.startswith('audio/'):
        return ContentAudio(audio=uri, format={'audio/wav': 'wav', 'audio/mpeg': 'mp3'}[artifact.media_type])
    if artifact.media_type == 'video/mp4':
        return ContentVideo(video=uri, format='mp4')
    return ContentDocument(document=uri, mime_type='application/pdf', filename='document.pdf')


def media_message(case: EvalCase, resolver: Callable[[MediaArtifact], bytes] | None):
    from inspect_ai.model import ContentText
    artifacts = case_media(case)
    if not artifacts:
        return case.input
    if resolver is None:
        raise ValueError('Bound media requires an explicit bytes resolver; refusing a text-only task')
    return [ContentText(text=case.input), *[to_inspect_content(a, resolver(a)) for a in artifacts]]


def verify_media_message(case: EvalCase, message) -> None:
    """Check retained original input, not arbitrary later model/tool attachments."""
    artifacts = case_media(case)
    if not artifacts:
        return
    from inspect_ai.model import ContentText
    content = message.content
    if (message.role != 'user' or not isinstance(content, list) or len(content) != len(artifacts) + 1
            or not isinstance(content[0], ContentText) or content[0].text != case.input):
        raise ValueError('Original Inspect media input is missing or changed')
    types = {'image': 'image/', 'audio': 'audio/', 'video': 'video/', 'document': 'application/pdf'}
    for artifact, item in zip(artifacts, content[1:]):
        if item.type not in types or not artifact.media_type.startswith(types[item.type]):
            raise ValueError('Inspect media type differs from bound artifact')
        expected_format = {'audio/wav': 'wav', 'audio/mpeg': 'mp3', 'video/mp4': 'mp4'}
        if item.type in {'audio', 'video'} and item.format != expected_format[artifact.media_type]:
            raise ValueError('Inspect media format differs from bound artifact')
        if item.type == 'document' and item.mime_type != artifact.media_type:
            raise ValueError('Inspect document MIME type differs from bound artifact')
        uri = getattr(item, item.type)
        prefix = f'data:{artifact.media_type};base64,'
        if not isinstance(uri, str) or not uri.startswith(prefix):
            raise ValueError('Resolve native log media attachments before importing bound media evidence')
        try:
            payload = base64.b64decode(uri[len(prefix):], validate=True)
        except ValueError as exc:
            raise ValueError('Invalid inline Inspect media encoding') from exc
        artifact.verify(payload)
