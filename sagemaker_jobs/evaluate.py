from __future__ import annotations

import json
from pathlib import Path
import tarfile


MODEL_INPUT = Path("/opt/ml/processing/model")
EVALUATION_OUTPUT = Path("/opt/ml/processing/evaluation/evaluation.json")


def _extract_model_artifact() -> Path:
    archives = list(MODEL_INPUT.glob("*.tar.gz")) + list(MODEL_INPUT.glob("*.tgz"))
    if archives:
        extracted = MODEL_INPUT / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archives[0], "r:gz") as archive:
            root = extracted.resolve()
            for member in archive.getmembers():
                if not (root / member.name).resolve().is_relative_to(root):
                    raise ValueError(f"Unsafe archive member: {member.name}")
            archive.extractall(extracted)
        return extracted
    return MODEL_INPUT


def main() -> None:
    model_root = _extract_model_artifact()
    gate_path = next(model_root.rglob("model_gate.json"), None)
    if gate_path is None:
        raise FileNotFoundError("model_gate.json was not found in the training artifact")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    evaluation = {
        "gate": gate,
        "regression_metrics": {
            "candidate_mae": {"value": gate.get("candidate_mae")},
            "hourly_baseline_mae": {"value": gate.get("hourly_baseline_mae")},
            "coverage_pct": {"value": gate.get("coverage_pct")},
        },
    }
    EVALUATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_OUTPUT.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print("Evaluation:", EVALUATION_OUTPUT)


if __name__ == "__main__":
    main()
