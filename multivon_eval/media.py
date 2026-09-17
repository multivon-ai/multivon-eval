"""Content bindings for caller-owned media; loading/storage/decoding stay upstream."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, replace

from .case import EvalCase
from .case_manifest import canonical_json, digest

MEDIA_KEY = 'multivon_media_v1'
MEDIA_TYPES = {'image/png': 'Image', 'image/jpeg': 'Image', 'image/webp': 'Image',
               'audio/wav': 'Sound', 'audio/mpeg': 'Sound', 'video/mp4': 'Video',
               'application/pdf': 'Text'}


@dataclass(frozen=True)
class MediaArtifact:
    """Immutable descriptor, not a media store or authenticity signature.

    Capture probes caller-supplied bytes using established parsers. Loading a
    descriptor verifies its structure/digest, not the existence of the bytes;
    call verify before use. Probe/provenance values are not authenticated.
    """
    _json: str

    def __post_init__(self):
        data = json.loads(self._json)
        if not isinstance(data, dict):
            raise TypeError('Media descriptor must be an object')
        claimed = data.pop('digest', None)
        if set(data) != {'schema', 'sha256', 'size_bytes', 'media_type', 'properties', 'probe', 'provenance'}:
            raise ValueError('Invalid media descriptor fields')
        if data['schema'] != 'multivon.media/v1' or digest(data) != claimed:
            raise ValueError('Invalid media descriptor schema/digest')
        sha = data['sha256']
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid media SHA-256')
        if type(data['size_bytes']) is not int or data['size_bytes'] <= 0:
            raise ValueError('Media size must be a positive integer')
        if data['media_type'] not in MEDIA_TYPES:
            raise ValueError('Unsupported media type')
        if not all(isinstance(data[k], dict) for k in ('properties', 'probe', 'provenance')):
            raise ValueError('Media properties/probe/provenance must be objects')
        from .media_probe import validate_properties
        validate_properties(data['media_type'], data['properties'])

    @classmethod
    def capture(cls, content: bytes, media_type: str, *, provenance: dict | None = None,
                max_bytes: int = 32 * 1024 * 1024) -> MediaArtifact:
        """Probe explicitly supplied bytes, without resolving any path or URL."""
        if not isinstance(content, bytes) or not content:
            raise ValueError('Supply nonempty immutable bytes')
        if type(max_bytes) is not int or max_bytes < 1 or len(content) > max_bytes:
            raise ValueError('Media exceeds the positive max_bytes limit')
        if media_type not in MEDIA_TYPES:
            raise ValueError('Unsupported media type')
        from .media_probe import probe_media
        properties, probe = probe_media(content, media_type)
        data = {'schema': 'multivon.media/v1', 'sha256': hashlib.sha256(content).hexdigest(),
                'size_bytes': len(content), 'media_type': media_type, 'properties': properties,
                'probe': probe, 'provenance': dict(provenance or {})}
        return cls.from_dict({**data, 'digest': digest(data)})

    @classmethod
    def from_dict(cls, data: dict) -> MediaArtifact:
        return cls(canonical_json(data))

    @property
    def data(self) -> dict:
        return json.loads(self._json)

    @property
    def id(self) -> str:
        # Include provenance so identical page/frame bytes can retain distinct roles.
        return 'urn:multivon:media:' + self.data['digest']

    @property
    def media_type(self) -> str:
        return self.data['media_type']

    def verify(self, content: bytes) -> bytes:
        """Check bytes and re-probe geometry/timing before use; return identical bytes."""
        if (not isinstance(content, bytes) or len(content) != self.data['size_bytes']
                or hashlib.sha256(content).hexdigest() != self.data['sha256']):
            raise ValueError('Media bytes do not match the bound content')
        from .media_probe import probe_media
        properties, _ = probe_media(content, self.media_type)
        if properties != self.data['properties']:
            raise ValueError('Media properties differ from the actual bytes/current parser')
        return content

    def data_uri(self, content: bytes) -> str:
        return f'data:{self.media_type};base64,' + base64.b64encode(self.verify(content)).decode('ascii')


def case_media(case: EvalCase) -> tuple[MediaArtifact, ...]:
    data = case.metadata.get(MEDIA_KEY, [])
    if not isinstance(data, list):
        raise TypeError('Bound media must be an ordered list of descriptors')
    artifacts = tuple(MediaArtifact.from_dict(item) for item in data)
    return artifacts


def with_media(case: EvalCase, *artifacts: MediaArtifact) -> EvalCase:
    """Return a case whose identity includes ordered media descriptors."""
    if MEDIA_KEY in case.metadata:
        raise ValueError('Case already has bound media; create an explicit case revision to replace it')
    if not artifacts:
        raise ValueError('Supply at least one media artifact')
    result = replace(case, metadata={**case.metadata, MEDIA_KEY: [a.data for a in artifacts]})
    case_media(result)
    return result
