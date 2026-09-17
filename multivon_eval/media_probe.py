"""Bounded metadata probes through Pillow, PDFium and PyAV; not a codec implementation."""
from __future__ import annotations

import io
import math
import threading
from importlib.metadata import version

# PDFium forbids concurrent calls even on different documents. Other users of
# PDFium in this process must coordinate externally or use isolated processes.
PDFIUM_LOCK = threading.RLock()


def validate_properties(mime: str, props: dict) -> None:
    fields = ({'width', 'height', 'orientation'} if mime.startswith('image/') else
              {'pages'} if mime == 'application/pdf' else
              {'duration_seconds', 'stream_index', 'width', 'height'} if mime.startswith('video/') else
              {'duration_seconds', 'stream_index', 'sample_rate', 'channels'})
    if set(props) != fields:
        raise ValueError('Media properties do not match its type')
    for name, value in props.items():
        if name == 'duration_seconds':
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError('Media duration must be positive/finite or unknown')
        elif type(value) is not int or value < (0 if name == 'stream_index' else 1):
            raise ValueError(f'Invalid media property {name}')
    if mime.startswith('image/') and props['orientation'] not in range(1, 9):
        raise ValueError('Invalid EXIF orientation')


def probe_media(content: bytes, mime: str) -> tuple[dict, dict]:
    if mime.startswith('image/'):
        from PIL import Image
        with Image.open(io.BytesIO(content)) as image:
            expected = {'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WEBP'}[mime]
            if image.format != expected or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Image encoding differs from MIME type or has multiple frames')
            width, height = image.size
            if width * height > 40_000_000:
                raise ValueError('Image exceeds 40 million pixel probe limit')
            orientation = image.getexif().get(274, 1)
            image.load()
        props = {'width': width, 'height': height, 'orientation': orientation}
        probe = {'library': 'Pillow', 'version': version('Pillow'), 'scope': 'single decoded image'}
    elif mime == 'application/pdf':
        import pypdfium2 as pdfium
        with PDFIUM_LOCK, pdfium.PdfDocument(content) as document:
            props = {'pages': len(document)}
        probe = {'library': 'pypdfium2', 'version': version('pypdfium2'),
                 'pdfium': str(pdfium.PDFIUM_INFO), 'scope': 'document open and page count; pages not rendered'}
    else:
        import av
        expected = {'audio/wav': 'wav', 'audio/mpeg': 'mp3', 'video/mp4': 'mp4'}[mime]
        with av.open(io.BytesIO(content), format=expected,
                     options={'protocol_whitelist': '', 'enable_drefs': '0', 'use_absolute_path': '0'}) as container:
            if expected not in container.format.name.split(','):
                raise ValueError('Media container differs from MIME type')
            kind = 'video' if mime.startswith('video/') else 'audio'
            streams = [s for s in container.streams if s.type == kind]
            if len(streams) != 1:
                raise ValueError('Select a file with exactly one primary stream of this media kind')
            stream = streams[0]
            duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                        and stream.time_base is not None else None)
            props = {'duration_seconds': duration, 'stream_index': stream.index}
            if kind == 'video':
                props.update(width=stream.width, height=stream.height)
            else:
                props.update(sample_rate=stream.sample_rate, channels=stream.channels)
        probe = {'library': 'av', 'version': version('av'),
                 'scope': 'container and primary stream metadata; packet decodability not verified'}
    validate_properties(mime, props)
    return props, probe
