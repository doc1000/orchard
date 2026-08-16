# Packaged Domain / Function taxonomy heads

These are AppWorld Phase 2B student heads trained on tool text (API name +
description + parameter docs + normalized keywords). They are an
**AppWorld-tool prior**. They will run on any document; they are **not** a
universal document classifier.

Orchard `transform` uses `title` + `text` only (no `app_name`, no LLM
keywords). That is a domain shift relative to the training text. Recorded
here so callers do not treat the default heads as a general document
classifier.

## Sources (export only)

| Packaged name | Source student (gitignored, not shipped) | SHA-256 |
|---|---|---|
| `domain` | `tool-tree-demo/artifacts/appworld/phase2b/classification/runtime/models/domain_taxonomy_v0_student.joblib` | `b927c99282099f322396b5671e58d2796d2f6c23ff8e1241f2bd6fe609b4021b` |
| `function` | `tool-tree-demo/artifacts/appworld/phase2b/classification/runtime/models/functional_taxonomy_v0_student.joblib` | `13fa57af1f0e18d568c36091a465499baa6f6e661dce877fad714c4b741d0114` |

Payloads are weight dicts `{weights, intercept, classes}` (not a full sklearn
pickle). Runtime loads `domain.npz` / `function.npz` and rebuilds
`LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000,
random_state=20260725)`. After export, Orchard does not read
`artifacts/appworld/` or `joblib`.

Label **sets** match `../domain.json` / `../function.json`. Label **order**
does not: students are sklearn-alphabetical; Orchard `label_order` is
definition order. Load remaps `predict_proba` columns onto packaged
`label_order`.

Features: `answerdotai/ModernBERT-base` revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`, 768-d, attention-mask mean pool,
`max_length=512`. Do not vendor `registry.json` or absolute `model_file`
paths.

Re-export with `python scripts/export_taxonomy_heads.py` from a checkout that
has the local students. Do not re-train heads in this phase.
