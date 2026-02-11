from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base_dir: Path, maybe_rel: str) -> str:
    p = Path(maybe_rel)
    return str(p if p.is_absolute() else (base_dir / p))


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    csv_path: str
    method_key: str


def load_runs_from_files(*json_paths: str | Path) -> List[RunSpec]:
    """
    Load and merge run specs from multiple files (ours/external/baselines).
    - resolves csv_path relative to each json file location
    - run_id must be unique across files
    """
    all_runs: list[RunSpec] = []
    seen: set[str] = set()

    for p in json_paths:
        p = Path(p)
        base_dir = p.parent
        raw = read_json(p)
        if not isinstance(raw, list):
            raise ValueError(f"{p} must contain a JSON list of runs.")

        for r in raw:
            run_id = r["run_id"]
            if run_id in seen:
                raise ValueError(f"Duplicate run_id '{run_id}' across run files.")
            seen.add(run_id)

            all_runs.append(
                RunSpec(
                    run_id=run_id,
                    label=r["label"],
                    csv_path=_resolve_path(base_dir, r["csv_path"]),
                    method_key=r.get("method_key", "unknown"),
                )
            )

    return all_runs


def load_methods(methods_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns: method_key -> metadata dict
    """
    raw = read_json(methods_path)
    methods = raw.get("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("methods.json must have top-level {'methods': {...}}")
    return methods


def load_benchmarks(benchmarks_path: str | Path) -> Dict[str, Any]:
    """
    Returns full dict from benchmarks.json
    """
    raw = read_json(benchmarks_path)
    if not isinstance(raw, dict):
        raise ValueError("benchmarks.json must be a dict.")
    return raw
