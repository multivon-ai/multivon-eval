"""Prepare the frozen CORD/pdfhell selection through upstream loaders/generators."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from decimal import ROUND_HALF_UP, Decimal
from importlib.metadata import version
from pathlib import Path

from datasets import Image, load_dataset
from pdfhell.generators import generate_case
from pdfhell.raster import pdfium_build, rasterize_pdf
from PIL import Image as PILImage

from multivon_eval import CaseManifest, EvalCase

CORD_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
FAMILIES = ("hidden_ocr_mismatch", "currency_mismatch_conversion")


def asset(path: Path, root: Path, media_type: str, *, limit: bool = True) -> dict:
    data = path.read_bytes()
    if limit and len(data) > 4_000_000:
        raise ValueError(f"Payload exceeds protocol: {path.name}")
    return {"path": str(path.relative_to(root)), "media_type": media_type,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def decimal_oracle(case) -> str:
    metadata = case.metadata
    if case.trap_family == "hidden_ocr_mismatch":
        amount = Decimal(str(metadata["visible_amount"]))
    else:
        amount = Decimal(str(metadata["eur_total"])) * Decimal(str(metadata["usd_rate"]))
    amount = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    expected = Decimal(case.expected_answer.removeprefix("$").replace(",", ""))
    if amount != expected:
        raise ValueError(f"Independent oracle disagrees with generator: {case.id}")
    return str(amount)


def prepare(root: Path, split: str) -> CaseManifest:
    root.mkdir(parents=True, exist_ok=False)
    assets = root / "assets"
    assets.mkdir()
    cases, excluded = [], []
    upstream_split, count = ("validation", 4) if split == "development" else ("test", 20)
    dataset = load_dataset("naver-clova-ix/cord-v2", revision=CORD_REVISION,
                           split=upstream_split, streaming=True).cast_column("image", Image(decode=False))
    for index, record in enumerate(dataset.take(count)):
        annotation = json.loads(record["ground_truth"])
        total = annotation.get("gt_parse", {}).get("total", {})
        expected = total.get("total_price") if isinstance(total, dict) else None
        if not isinstance(expected, str) or not expected.strip():
            excluded.append({"source": "cord", "row": index, "reason": "missing/non-scalar total"})
            continue
        source_id = f"cord-{upstream_split}-{index}"
        image_bytes = record["image"]["bytes"]
        with PILImage.open(io.BytesIO(image_bytes)) as image:
            original_media = PILImage.MIME[image.format]
            suffix = {"image/png": ".png", "image/jpeg": ".jpg"}[original_media]
            original = assets / f"{source_id}.original{suffix}"
            original.write_bytes(image_bytes)
            rendered = image.convert("RGB")
            rendered.thumbnail((1536, 1536), PILImage.Resampling.LANCZOS)
            path = assets / f"{source_id}.input.png"
            rendered.save(path)
        transcript = "\n".join(" ".join(word["text"] for word in line["words"])
                               for line in annotation["valid_line"])
        if len(transcript) > 32_000:
            raise ValueError("CORD transcript exceeds protocol")
        text_path = assets / f"{source_id}.txt"
        text_path.write_text(transcript, encoding="utf-8")
        for modality, file, media in [("image", path, "image/png"),
                                      ("oracle-text", text_path, "text/plain")]:
            cases.append(EvalCase(
                "Read this receipt and call post_entry once with its final total, not cash "
                "tendered or change. Transcribe the amount verbatim, preserving separators. "
                'Set currency to "UNSPECIFIED"; do not infer it. Do not merely describe an entry.',
                case_id=f"{source_id}-{modality}", source_id=source_id,
                tags=["cord", modality], metadata={"family": "cord", "modality": modality,
                    "asset": asset(file, root, media), "expected_amount": expected,
                    "expected_currency": "UNSPECIFIED", "upstream_meta": annotation["meta"],
                    "original_image": asset(original, root, original_media, limit=False),
                    "image_preprocessing": "RGB, Lanczos thumbnail 1536x1536, no upscale, PNG",
                    "upstream_row": index, "upstream_split": upstream_split,
                    "annotation_sha256": hashlib.sha256(record["ground_truth"].encode()).hexdigest()},
            ))
    seeds = range(42, 44) if split == "development" else range(91700, 91710)
    for family in FAMILIES:
        for seed in seeds:
            data, fixture = generate_case(family, seed)
            amount = decimal_oracle(fixture)
            path = assets / f"{fixture.id}.pdf"
            path.write_bytes(data)
            pages = rasterize_pdf(path, dpi=150)
            if len(pages) != 1:
                raise ValueError("Protocol requires single-page inputs")
            for modality, file, media in [("pdf", path, "application/pdf"),
                                          ("pixels", pages[0], "image/png")]:
                rule = ("Use the visible TOTAL DUE, even if a hidden text layer conflicts."
                        if family == "hidden_ocr_mismatch" else
                        "Use the settlement total in USD, applying the printed conversion rate "
                        "and rounding to the nearest cent, with ties rounded up.")
                cases.append(EvalCase(
                    "Read the invoice and call post_entry once. " + rule +
                    ' Use exactly two decimal places without grouping in amount and currency "USD". '
                    "Do not merely describe an entry.", case_id=f"{fixture.id}-{modality}",
                    source_id=fixture.id, tags=[family, modality], metadata={
                        "family": family, "modality": modality, "seed": seed,
                        "asset": asset(file, root, media), "source_pdf": asset(path, root, "application/pdf"),
                        "expected_amount": amount, "expected_currency": "USD",
                        "raster_dpi": 150 if modality == "pixels" else None,
                        "pdfium_build": pdfium_build() if modality == "pixels" else None},
                ))
    manifest = CaseManifest("document-ledger-v1", cases,
        splits={split: [case.case_id for case in cases]}, provenance={
            "protocol": "DOCUMENT_PROTOCOL.md v1", "cord_repository": "naver-clova-ix/cord-v2",
            "cord_revision": CORD_REVISION, "cord_license": "CC-BY-4.0",
            "cord_attribution": "NAVER Corp., CORD: A Consolidated Receipt Dataset for Post-OCR Parsing",
            "pdfhell_revision": "16d184b", "exclusions": excluded,
            "versions": {name: version(name) for name in ("datasets", "pdfhell", "reportlab", "Pillow", "pypdfium2")}})
    manifest.save(root / "manifest.json")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "held-out"], required=True)
    args = parser.parse_args()
    result = prepare(args.output, args.split)
    print(json.dumps({"cases": len(result.cases), "manifest_digest": result.digest}))
