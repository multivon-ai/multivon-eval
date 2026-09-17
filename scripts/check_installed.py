"""Offline checks against an installed wheel, never the source checkout.

Run from a clean environment outside the checkout; --bare also checks that
optional integration packages and pytest were not pulled into the base install.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import socket
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bare", action="store_true")
    args = parser.parse_args()

    def offline(*_args, **_kwargs):
        raise AssertionError("Installed-package checks must not open network connections")
    socket.socket.connect = offline
    socket.socket.connect_ex = offline

    import multivon_eval as m
    from multivon_eval.report_schema import report_schema, validate_report

    root = Path(__file__).resolve().parents[1]
    package = Path(m.__file__).resolve()
    assert not package.is_relative_to(root), "Source checkout shadows the installed package"
    assert m.__version__ == version("multivon-eval")
    if args.bare:
        for name in ("pytest", "datasets", "inspect_ai", "gymnasium", "sklearn", "scipy",
                     "PIL", "pypdfium2", "av", "litellm", "torch"):
            assert importlib.util.find_spec(name) is None, f"Unexpected optional dependency: {name}"
        try:
            m.assert_evaluators()
        except ImportError as exc:
            assert "multivon-eval[pytest]" in str(exc)
        else:
            raise AssertionError("Missing pytest must raise an actionable error")

    assert report_schema()["$schema"].endswith("2020-12/schema")
    suite = m.EvalSuite("installed-wheel").add_evaluators(m.ExactMatch())
    report = suite.run_on_cases([(m.EvalCase("q", "a"), "a")], verbose=False)
    payload = json.loads(report.to_json())
    validate_report(payload)
    assert m.EvalReport.from_dict(payload).passed == 1
    for filename in ("legacy-v1.json", "published-0.18.0.json"):
        fixture = json.loads((root / "tests" / "fixtures" / "reports" / filename).read_text())
        restored = m.EvalReport.from_dict(fixture)
        assert restored.total == 3 and restored.passed == 1
    print(f"Installed {m.__version__}: base API, schema resource and report migrations passed")


if __name__ == "__main__":
    main()
