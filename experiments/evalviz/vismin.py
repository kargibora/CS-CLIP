from __future__ import annotations
from typing import Any, Optional, List, Tuple
import json
import ast


def parse_vismin_value(value_raw: Any) -> Optional[List[Tuple[str, str, float]]]:
    try:
        if isinstance(value_raw, str):
            try:
                nested = json.loads(value_raw.replace("'", '"'))
            except Exception:
                nested = ast.literal_eval(value_raw)
        elif isinstance(value_raw, dict):
            nested = value_raw
        else:
            return None

        if not isinstance(nested, dict) or not nested:
            return None

        first_val = next(iter(nested.values()), None)
        if not isinstance(first_val, dict):
            return None
        if not any(k in first_val for k in ["text", "image", "group"]):
            return None

        expanded: List[Tuple[str, str, float]] = []
        for category, metrics in nested.items():
            if not isinstance(metrics, dict):
                continue
            for metric_type, metric_value in metrics.items():
                if metric_type == "count":
                    continue
                expanded.append((str(category), f"{metric_type}_contrastive_accuracy", float(metric_value)))

        return expanded if expanded else None
    except Exception:
        return None
