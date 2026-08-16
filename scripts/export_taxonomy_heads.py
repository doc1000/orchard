"""One-time export of AppWorld Phase 2B student heads into portable arrays.

Reads the local gitignored joblibs under tool-tree-demo/artifacts/appworld/,
verifies SHA-256s, and writes orchard-owned npz + sidecar JSON. Runtime does
not depend on those joblibs, joblib, or artifacts/appworld/ after export.

Do not git-add the raw .joblib files or registry.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

ORCHARD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ORCHARD_ROOT.parent
SRC = ORCHARD_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orchard.backends.modernbert import (  # noqa: E402
    FEATURE_CONFIG,
    MODERNBERT_MODEL_ID,
    MODERNBERT_REVISION,
)
from orchard.backends.taxonomy_heads import (  # noqa: E402
    HEAD_HYPERPARAMS,
    HEAD_SCHEMA_VERSION,
    SOURCE_STUDENT_SHA256,
)
from orchard.taxonomy import TaxonomyModel  # noqa: E402

DEFAULT_MODELS_DIR = (
    WORKSPACE_ROOT
    / "tool-tree-demo"
    / "artifacts"
    / "appworld"
    / "phase2b"
    / "classification"
    / "runtime"
    / "models"
)
STUDENT_FILES = {
    "domain": "domain_taxonomy_v0_student.joblib",
    "function": "functional_taxonomy_v0_student.joblib",
}
HEADS_DIR = ORCHARD_ROOT / "src" / "orchard" / "assets" / "taxonomies" / "heads"
TAXONOMY_DIR = ORCHARD_ROOT / "src" / "orchard" / "assets" / "taxonomies"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_student(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"student joblib missing: {path}")
    digest = _sha256_file(path)
    if digest != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {digest}"
        )
    return digest


def _definition_sha256(name: str) -> str:
    path = TAXONOMY_DIR / f"{name}.json"
    return _sha256_file(path)


def _export_one(name: str, models_dir: Path, out_dir: Path) -> dict[str, Any]:
    student_path = models_dir / STUDENT_FILES[name]
    digest = _verify_student(student_path, SOURCE_STUDENT_SHA256[name])
    payload = joblib.load(student_path)
    if set(payload) != {"weights", "intercept", "classes"}:
        raise ValueError(f"{student_path.name} is not a {{weights, intercept, classes}} dict")
    weights = np.asarray(payload["weights"], dtype=np.float64)
    intercept = np.asarray(payload["intercept"], dtype=np.float64)
    classes = [str(label) for label in np.asarray(payload["classes"]).tolist()]
    taxonomy = TaxonomyModel.load_default(name)
    if set(classes) != set(taxonomy.label_order):
        raise ValueError(
            f"{name} student class set does not match Orchard label_order"
        )
    expected_shape = (len(taxonomy.label_order), int(FEATURE_CONFIG["dimensions"]))
    if weights.shape != expected_shape:
        raise ValueError(f"{name} weights shape {weights.shape} != {expected_shape}")
    if intercept.shape != (len(taxonomy.label_order),):
        raise ValueError(f"{name} intercept shape {intercept.shape} is invalid")

    npz_path = out_dir / f"{name}.npz"
    np.savez_compressed(
        npz_path,
        coef_=weights,
        intercept_=intercept,
        classes_=np.asarray(classes),
    )
    sidecar = {
        "schema_version": HEAD_SCHEMA_VERSION,
        "taxonomy_name": taxonomy.name,
        "taxonomy_version": taxonomy.taxonomy_version,
        "classes_": classes,
        "classes_order": "sklearn_alphabetical_student",
        "label_order": list(taxonomy.label_order),
        "remap": (
            "stored classes_ are student/sklearn-alphabetical; "
            "runtime remaps predict_proba columns onto packaged label_order"
        ),
        "feature_model": {
            "id": MODERNBERT_MODEL_ID,
            "revision": MODERNBERT_REVISION,
            "pooling": FEATURE_CONFIG["pooling"],
            "max_length": FEATURE_CONFIG["max_length"],
            "dimensions": FEATURE_CONFIG["dimensions"],
            "batch_size": FEATURE_CONFIG["batch_size"],
        },
        "logistic": dict(HEAD_HYPERPARAMS),
        "source_student": STUDENT_FILES[name],
        "source_student_sha256": digest,
        "taxonomy_definition_sha256": _definition_sha256(name),
        "export_date": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "estimator": LogisticRegression.__module__ + ".LogisticRegression",
    }
    sidecar_path = out_dir / f"{name}.json"
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "name": name,
        "npz": str(npz_path),
        "sidecar": str(sidecar_path),
        "n_classes": len(classes),
        "coef_shape": list(weights.shape),
        "source_student_sha256": digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing the two student joblibs (read-only)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HEADS_DIR,
        help="Directory for portable npz + sidecar JSON",
    )
    args = parser.parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_export_one(name, args.models_dir, out_dir) for name in STUDENT_FILES]
    for row in summaries:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
