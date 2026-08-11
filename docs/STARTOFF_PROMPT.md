# Agent startoff prompt

Copy everything below the line into a new agent chat (repo root = this `orchard` repository).

---

You are implementing the standalone Orchard library in this repository.

## Read first (in order)

1. `docs/ORCHARD_EXTRACTION_PLAN.md` — **execution plan; follow phases**
2. `docs/ORCHARD_ARCHITECTURE.md` — architectural specification
3. `docs/RELEASE_GATE.md` — definition of done
4. `CODE_MANIFEST.md` — create/update per the plan (required navigation artifact)

Explicitly **ignore** `docs/archive/ORCHARD_REFACTOR_DECISIONS.md` and any older fused/single-tree/AppWorld-consumer language that conflicts with the extraction plan.

Reference implementation (read-only; do not refactor): `../tool-tree-demo`.

## Current assignment

Start at **Phase 0** of `docs/ORCHARD_EXTRACTION_PLAN.md` unless `CODE_MANIFEST.md` and Phase 0 outputs already exist and pass that phase’s release gate — then continue with the next incomplete phase.

Complete **one phase at a time**. Stop at the phase release gate and summarize what passed before starting the next phase unless asked to continue.

## Hard constraints

- Multi-tree `Orchard` by default; one tree per configured taxonomy; no taxonomies → `semantic` tree only.
- No fused/mixed Domain+Function+semantic default tree.
- Public construction verb is `build` (not `fit_transform`).
- Extract working algorithms largely as-is; do not redesign numerics.
- No AppWorld integration, eval, README examples, or release proof.
- Contrastive labeling is out of scope (future only).
- Labels never mutate linkage.
- Update `CODE_MANIFEST.md` at the end of every phase.
- Prefer generic fixtures over demo-specific data.

## Working style

- Explain major changes briefly before making them.
- Prefer small, reviewable commits only if the user asks for commits.
- Keep the README short; richer docs come in Phase 4.
- When stuck on provenance, use the source map appendix in the extraction plan.
