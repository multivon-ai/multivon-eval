"""Render a cited PDF page with the existing PDFium/Pillow stack."""
from __future__ import annotations

import io
import math
from contextlib import closing
from importlib.metadata import version

from .media import MediaArtifact
from .media_probe import PDFIUM_LOCK


def render_pdf_page(artifact: MediaArtifact, content: bytes, *, page: int, scale: float = 1.5):
    """Return (page artifact, PNG bytes); page numbers are one-based.

    Scale is pixels per PDF point. Coordinates on the returned PNG refer to its
    final raster, not PDF points. The parent bytes and rendering recipe remain
    in provenance. This does not authenticate the document or its text layer.
    """
    if artifact.media_type != 'application/pdf':
        raise ValueError('Page rendering requires a PDF artifact')
    artifact.verify(content)
    if type(page) is not int or page < 1:
        raise ValueError('Page must be a positive one-based integer')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be positive and finite')
    import pypdfium2 as pdfium
    with PDFIUM_LOCK, pdfium.PdfDocument(content) as document:
        if page > len(document):
            raise ValueError('Page exceeds document length')
        with closing(document[page - 1]) as native_page:
            width, height = native_page.get_size()
            if math.ceil(width * scale) * math.ceil(height * scale) > 40_000_000:
                raise ValueError('Rendered page exceeds 40 million pixel limit')
            bitmap = native_page.render(scale=scale)
            try:
                output = io.BytesIO()
                image = bitmap.to_pil()
                image.save(output, format='PNG')
                image.close()
            finally:
                bitmap.close()
    data = output.getvalue()
    derived = MediaArtifact.capture(data, 'image/png', provenance={
        'operation': 'pdf_page_render', 'parent': artifact.data, 'page': page,
        'renderer': 'pypdfium2', 'version': version('pypdfium2'), 'pdfium': str(pdfium.PDFIUM_INFO),
        'scale_pixels_per_point': scale, 'form_rendering': False,
        'limits': 'PDFium default page rendering; form widgets are not initialized'})
    return derived, data
