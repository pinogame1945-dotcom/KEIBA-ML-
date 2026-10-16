"""One public Python API for the existing frozen race-ID to V3 pipeline.

Internal research models and contracts are unchanged. Missing real upstream
sources must remain BLOCKED; this package does not make surrogate predictions.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "research" / "run_github_race_ids_v1.py"


def analyze(race_ids: list[str], manifest_path: str | Path) -> dict:
    source = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("keiba_frozen_runtime_engine", ENGINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("EXISTING_FROZEN_ENGINE_NOT_FOUND")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    return engine.run(list(race_ids), manifest, source.parent)


def analyze_to_json(race_ids: list[str], manifest_path: str | Path,
                    output_path: str | Path) -> dict:
    result = analyze(race_ids, manifest_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return result
