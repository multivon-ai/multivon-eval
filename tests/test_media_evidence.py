"""Native media bytes, selectors, actual Inspect logs and missing-media failures."""
import copy
import io
import json
import wave

import pytest

pytest.importorskip('PIL')
pytest.importorskip('av')
pytest.importorskip('pypdfium2')

from PIL import Image, UnidentifiedImageError

from multivon_eval import CaseManifest, EvalCase, ExactMatch
from multivon_eval.grounding import annotation, grounded_result, validate_annotation
from multivon_eval.media import MediaArtifact, case_media, with_media
from multivon_eval.media_pdf import render_pdf_page


def png(color='white', width=64, height=48):
    stream = io.BytesIO()
    Image.new('RGB', (width, height), color).save(stream, format='PNG')
    return stream.getvalue()


def native_media():
    import av
    import pypdfium2 as pdfium
    image = png()
    audio = io.BytesIO()
    with wave.open(audio, 'wb') as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(8000)
        file.writeframes(bytes(16000))
    pdf = io.BytesIO()
    with pdfium.PdfDocument.new() as document:
        document.new_page(120, 100)
        document.save(pdf)
    video = io.BytesIO()
    with av.open(video, 'w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for _ in range(4):
            frame = av.VideoFrame(64, 48, 'rgb24')
            frame.planes[0].update(bytes([120]) * frame.planes[0].buffer_size)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return [(image, 'image/png'), (audio.getvalue(), 'audio/wav'),
            (video.getvalue(), 'video/mp4'), (pdf.getvalue(), 'application/pdf')]


@pytest.fixture
def media():
    return [(MediaArtifact.capture(data, mime), data) for data, mime in native_media()]


def test_all_native_types_probe_and_verify(media):
    assert [a.media_type for a, _ in media] == ['image/png', 'audio/wav', 'video/mp4', 'application/pdf']
    for artifact, data in media:
        assert artifact.verify(data) is data
        assert MediaArtifact.from_dict(artifact.data) == artifact
        with pytest.raises(ValueError, match='bound content'):
            artifact.verify(data + b'changed')
    assert media[1][0].data['properties']['duration_seconds'] == 1
    assert media[2][0].data['properties']['duration_seconds'] == 1


def test_mutation_and_content_change_cannot_preserve_case_identity():
    data = png()
    artifact = MediaArtifact.capture(data, 'image/png')
    descriptor = artifact.data
    descriptor['properties']['width'] = 1
    with pytest.raises(ValueError, match='digest'):
        MediaArtifact.from_dict(descriptor)
    original = EvalCase('What is shown?', case_id='image')
    case = with_media(original, artifact)
    changed = with_media(original, MediaArtifact.capture(png('black'), 'image/png'))
    assert case.identity()[0] == changed.identity()[0]
    assert case.identity()[1] != changed.identity()[1]
    manifest = CaseManifest('media', [case])
    assert case_media(manifest.cases[0])[0] == artifact
    assert original.metadata == {}


def test_corrupt_mime_and_bounded_inputs_reject():
    with pytest.raises(ValueError, match='MIME'):
        MediaArtifact.capture(png(), 'image/jpeg')
    with pytest.raises(ValueError, match='max_bytes'):
        MediaArtifact.capture(png(), 'image/png', max_bytes=1)
    with pytest.raises(UnidentifiedImageError):
        MediaArtifact.capture(b'not a media file', 'image/png')


def test_pdf_render_retains_parent_page_and_native_dimensions(media):
    parent, data = media[3]
    page, rendered = render_pdf_page(parent, data, page=1, scale=2)
    assert page.data['properties']['width'] == 240
    assert page.data['properties']['height'] == 200
    assert page.data['provenance']['parent'] == parent.data
    assert page.data['provenance']['page'] == 1
    page.verify(rendered)
    with pytest.raises(ValueError, match='document length'):
        render_pdf_page(parent, data, page=2)
    with pytest.raises(ValueError, match='pixel limit'):
        render_pdf_page(parent, data, page=1, scale=10000)


def test_standard_annotations_and_unknown_verdict(media):
    image, audio, video, document = [a for a, _ in media]
    refs = [annotation(image, 'White pixels', region=(0, 0, 64, 48)),
            annotation(audio, 'Silence interval', time=(0, 0.5)),
            annotation(video, 'Gray frames', time=(0.25, 1)), annotation(document, 'Document exists')]
    for ref in refs:
        validate_annotation(json.loads(json.dumps(ref)), [image, audio, video, document])
    result = grounded_result('fixture', None, 'No semantic oracle', annotations=refs, artifacts=[image, audio, video, document])
    assert result.metadata['skipped'] and not result.passed
    refs[0]['body']['value'] = 'mutated'
    assert result.metadata['annotations'][0]['body']['value'] == 'White pixels'
    with pytest.raises(ValueError, match='at least one'):
        grounded_result('fixture', True, 'missing refs', annotations=[], artifacts=[image])


@pytest.mark.parametrize('region', [(-1,0,1,1), (0,0,0,1), (0,0,65,48), (63,47,2,2), (True,0,1,1)])
def test_out_of_bounds_regions_are_not_clipped(region):
    artifact = MediaArtifact.capture(png(), 'image/png')
    with pytest.raises(ValueError):
        annotation(artifact, 'claim', region=region)


@pytest.mark.parametrize('span', [(-1,0.5), (0.5,0.5), (0,1.1), (float('nan'),1), (True,1)])
def test_invalid_timestamps_reject(media, span):
    with pytest.raises(ValueError):
        annotation(media[1][0], 'claim', time=span)


def test_selector_ambiguity_and_external_targets_reject(media):
    artifact = media[0][0]
    ref = annotation(artifact, 'claim', region=(0,0,10,10))
    for fragment in ['xywh=pixel:0,0,10,10&xywh=pixel:2,2,3,3', 'xywh=percent:0,0,10,10',
                     't=npt:0,1', 'track=1']:
        changed = copy.deepcopy(ref)
        changed['target']['selector']['value'] = fragment
        with pytest.raises(ValueError):
            validate_annotation(changed, [artifact])
    ref['target']['source'] = 'https://untrusted.example/image.png'
    with pytest.raises(ValueError, match='known content'):
        validate_annotation(ref, [artifact])


def test_explicit_resolver_never_silently_drops_or_replaces_media(media):
    pytest.importorskip('inspect_ai')
    from multivon_eval.integrations.inspect import to_inspect_dataset
    case = with_media(EvalCase('Describe.', case_id='media'), *[a for a, _ in media])
    manifest = CaseManifest('native media', [case])
    with pytest.raises(ValueError, match='explicit bytes resolver'):
        to_inspect_dataset(manifest)
    with pytest.raises(ValueError, match='bound content'):
        to_inspect_dataset(manifest, media_resolver=lambda _: b'wrong content')
    content = {a.id: b for a, b in media}
    samples = to_inspect_dataset(manifest, media_resolver=lambda a: content[a.id])
    assert [c.type for c in samples[0].input[-1].content] == ['text','image','audio','video','document']


def test_native_inspect_run_and_resolved_log_roundtrip(media, tmp_path):
    pytest.importorskip('inspect_ai')
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import generate

    from multivon_eval.integrations.inspect import (
        as_inspect_scorer,
        from_inspect_log,
        to_inspect_dataset,
    )
    case = with_media(EvalCase('Describe.', 'ok', case_id='media', source_id='fixture'), *[a for a, _ in media])
    content = {a.id: b for a, b in media}
    task = Task(dataset=to_inspect_dataset(CaseManifest('media', [case]), media_resolver=lambda a: content[a.id]),
                solver=generate(), scorer=as_inspect_scorer(ExactMatch()))
    log = inspect_eval(task, model='mockllm/model', model_args={'custom_outputs': [ModelOutput.from_content('fixture', 'ok')]},
                       log_dir=str(tmp_path), display='none', log_images=True)[0]
    assert log.status == 'success', log.error
    loaded = read_eval_log(log.location, resolve_attachments=True)
    report = from_inspect_log(loaded)
    assert report.passed == 1
    assert report.case_results[0].trials[0].data['case']['metadata'] == case.metadata
    changed = copy.deepcopy(loaded)
    changed.samples[0].messages[0].content[1].image = media[0][0].data_uri(png()) + 'wrong'
    with pytest.raises(ValueError):
        from_inspect_log(changed)


def test_repeated_identical_pages_retain_roles_and_input_order():
    data = png()
    a = MediaArtifact.capture(data, 'image/png', provenance={'page': 1})
    b = MediaArtifact.capture(data, 'image/png', provenance={'page': 2})
    assert a.data['sha256'] == b.data['sha256'] and a.id != b.id
    case = with_media(EvalCase('Compare pages.'), a, b, a)
    assert case_media(case) == (a, b, a)
    first, second = annotation(a, 'Page one'), annotation(b, 'Page two')
    assert validate_annotation(first, [a, b]) == a
    assert validate_annotation(second, [a, b]) == b


def test_huggingface_native_image_bytes_feed_existing_manifest_bridge():
    datasets = pytest.importorskip('datasets')
    from multivon_eval.integrations.huggingface import from_huggingface
    data = png()
    dataset = datasets.Dataset.from_dict({'id': ['source-1'], 'image': [data]}).cast_column(
        'image', datasets.Image(decode=False))
    manifest = from_huggingface(dataset, name='native HF image', record_to_case=lambda row:
        with_media(EvalCase('Describe.', case_id=row['id'], source_id=row['id']),
                   MediaArtifact.capture(row['image']['bytes'], 'image/png')))
    assert case_media(manifest.cases[0])[0].verify(data) == data


def test_rehashed_false_geometry_cannot_verify_against_real_bytes():
    from multivon_eval.case_manifest import digest
    data = png()
    descriptor = MediaArtifact.capture(data, 'image/png').data
    descriptor.pop('digest')
    descriptor['properties']['width'] = 1000
    forged = MediaArtifact.from_dict({**descriptor, 'digest': digest(descriptor)})
    with pytest.raises(ValueError, match='properties differ'):
        forged.verify(data)


@pytest.mark.parametrize('format,mime', [('JPEG','image/jpeg'),('WEBP','image/webp')])
def test_other_supported_still_image_encodings(format, mime):
    output = io.BytesIO()
    Image.new('RGB', (32,24), 'white').save(output, format=format)
    artifact = MediaArtifact.capture(output.getvalue(), mime)
    artifact.verify(output.getvalue())
    assert validate_annotation(annotation(artifact, 'whole image', region=(0,0,32,24)), [artifact]) == artifact


def test_orientation_and_video_regions_require_explicit_normalized_still_image(media):
    output = io.BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new('RGB', (32,24), 'white').save(output, format='JPEG', exif=exif)
    artifact = MediaArtifact.capture(output.getvalue(), 'image/jpeg')
    for item in [artifact, media[2][0]]:
        with pytest.raises(ValueError, match='orientation-1 still image'):
            annotation(item, 'region', region=(0,0,1,1))


def test_native_mp3_probe_and_inspect_format():
    import av
    output = io.BytesIO()
    with av.open(output, 'w', format='mp3') as container:
        stream = container.add_stream('libmp3lame', rate=8000)
        stream.layout = 'mono'
        frame = av.AudioFrame(format='s16p', layout='mono', samples=8000)
        frame.sample_rate = 8000
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    artifact = MediaArtifact.capture(output.getvalue(), 'audio/mpeg')
    artifact.verify(output.getvalue())
    assert artifact.data['properties']['duration_seconds'] > 0
    pytest.importorskip('inspect_ai')
    from multivon_eval.integrations.inspect_media import to_inspect_content
    assert to_inspect_content(artifact, output.getvalue()).format == 'mp3'
