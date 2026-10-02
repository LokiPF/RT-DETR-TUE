# Practical Corruption Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace strict Python implementation introspection with the existing three-member corruption interface and document that changed corruption code requires a new output folder.

**Architecture:** Snapshot and validate only `name`, the six ordered `Severity` values, and the bound `apply(image, level)` operation. Provenance and reports retain the corruption name and severity table while all existing image, checkpoint, runtime, cache-integrity, atomic-publication, numerical, and detector checks remain unchanged.

**Tech Stack:** Python 3.11, Pillow, PyTorch, JSON, pytest, Markdown

---

## File structure

| File | Responsibility in this change |
|---|---|
| `differential_uncertainty/cli.py` | Remove plugin source/module/dependency/state inspection and terminal plugin revalidation; retain interface validation and name/severity provenance. |
| `differential_uncertainty/reporting.py` | Validate and render the smaller corruption provenance mapping. |
| `tests/differential_uncertainty/test_pipeline.py` | Replace strict-identity regressions with the practical interface, resume-boundary, and terminal-order tests. |
| `tests/differential_uncertainty/test_reporting.py` | Check reports contain the practical corruption identity and no strict implementation claims. |
| `README.md` | Explain the new-output-folder rule for changed corruption code. |
| `docs/clean-branch-verification.md` | Record the exact tested commit, test result, and retained repository counts. |

### Task 1: Simplify corruption snapshots and provenance

**Files:**
- Modify: `differential_uncertainty/cli.py:109-447,470-496,1122-1236`
- Modify: `differential_uncertainty/reporting.py:334-401,892-896,1061-1066`
- Modify: `tests/differential_uncertainty/test_pipeline.py:36-260,344-721,801-839,2325-2836`
- Modify: `tests/differential_uncertainty/test_reporting.py`

- [ ] **Step 1: Write the failing practical-interface tests**

Retain the existing end-to-end custom-corruption test, but make its plugin implement only the public contract:

```python
class ContrastCorruption:
    name = "contrast"
    severities = tuple(Severity(level, float(level)) for level in range(6))

    def apply(self, image, level):
        return ImageEnhance.Contrast(image).enhance(1.0 + level * 0.1)
```

Add or adapt assertions proving:

```python
assert set(summary["provenance"]["corruption"]) == {"name", "severities"}
assert summary["provenance"]["corruption"]["name"] == "contrast"
```

Retain tests that reject an invalid name, invalid severity table, changed name, or changed
severity table before cache reuse. Add a terminal-order assertion that `_final_audit` never
re-reads or revalidates plugin implementation state.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_pipeline.py \
  tests/differential_uncertainty/test_reporting.py \
  -q
```

Expected: practical plugins fail because the current code requires
`implementation_sha256`, `behavior_state`, and `dependency_sha256`, and reporting still
requires the `implementation` mapping.

- [ ] **Step 3: Remove strict implementation inspection**

Delete `_normalized_source`, `_strict_sha256`, `_canonical_plugin_state`,
`_module_file_snapshot`, `_declared_dependencies`, `_referenced_dependency_modules`, and
`_corruption_implementation`, then remove their now-unused imports.

Reduce the snapshot to:

```python
@dataclass(frozen=True)
class _CorruptionSnapshot:
    name: str
    severities: tuple[Severity, ...]
    _apply: Callable

    def apply(self, image, level):
        return self._apply(image, level)
```

Keep all existing name and severity validation in `_snapshot_corruption`, keep the callable
check for `apply`, and return:

```python
return _CorruptionSnapshot(name, tuple(frozen_severities), apply)
```

Do not add a replacement implementation digest, dependency walker, terminal mutation check,
or adapter system.

- [ ] **Step 4: Reduce corruption provenance and terminal audit**

Change `_provenance` to publish exactly:

```python
"corruption": {
    "name": corruption.name,
    "severities": [
        {"level": severity.level, "parameter": severity.parameter}
        for severity in corruption.severities
    ],
},
```

Remove `corruption.revalidate()` from `_final_audit`. Preserve the final sequence:

```text
full input-image validation -> artifact terminal sweep -> final lightweight image signatures
```

No plugin inspection may occur after extraction/report generation.

- [ ] **Step 5: Reduce report schema and wording**

In `reporting.py`, require the corruption mapping to contain exactly `name` and `severities`.
Delete validation and rendering of module, qualname, source/build SHA, behavior state, module
files, and dependency hashes. Replace the reproducibility paragraph with:

```text
The corruption was {corruption_name}, with severities {severity_parameters}. If corruption
code or hidden settings change, use a new output folder rather than resuming this run.
```

Retain the runtime library versions and checkpoint SHA in the same paragraph.

- [ ] **Step 6: Remove obsolete strict-identity tests and verify GREEN**

Remove tests dedicated solely to source availability, implementation/build SHA, behavior
state, dependency-module hashing, module mutation, or in-memory callable replacement. Remove
their test-only plugin declarations and imports. Do not remove tests for the public corruption
contract, changed name/severity, input fingerprints, checkpoint/runtime mismatch, artifact
integrity, atomic publication, or terminal input ordering.

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest \
  tests/differential_uncertainty/test_corruptions.py \
  tests/differential_uncertainty/test_pipeline.py \
  tests/differential_uncertainty/test_reporting.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Run the complete suite and commit code/tests**

Run:

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q -rs
/home/yuchen/miniconda3/envs/UE/bin/python -m compileall -q differential_uncertainty src tests/differential_uncertainty
git diff --check
```

Confirm the repository-surface guard still reports exactly 43 Python files. Commit only code
and tests:

```bash
git add differential_uncertainty/cli.py differential_uncertainty/reporting.py \
  tests/differential_uncertainty/test_pipeline.py \
  tests/differential_uncertainty/test_reporting.py
git commit -m "Simplify corruption plugin provenance"
```

### Task 2: Update user guidance and verification evidence

**Files:**
- Modify: `README.md:77-133`
- Modify: `docs/clean-branch-verification.md`

- [ ] **Step 1: Replace strict plugin documentation**

Document these exact rules in `README.md`:

```text
A corruption provides a name, six ordered severity values, and apply(image, level).
Resume assumes the same corruption code and hidden settings. If either changes, use a new
output folder. The workflow does not inspect arbitrary Python dependencies or in-memory state.
```

Keep the existing instructions for manifests, checkpoint verification, runtime controls,
outputs, AUROC, Spearman correlation, and report interpretation.

- [ ] **Step 2: Refresh the verification record from the tested code commit**

Record the exact Task 1 code commit hash, complete-suite pass count and duration, exact Python
file count, and exact retained line count. Remove claims about module/build/state/dependency
hashes and terminal plugin mutation detection. Keep verified claims for image, checkpoint,
runtime, cache, detector parity, report reconciliation, and the official checkpoint.

- [ ] **Step 3: Verify documentation and commit**

Run:

```bash
rg -n "implementation_sha256|behavior_state|dependency_sha256|complete corruption implementation|plugin or module changed" README.md docs/clean-branch-verification.md
git diff --check
git status --short
```

Expected: the strict-identity search has no claims in the two current user documents; only the
two documentation files are uncommitted. Commit:

```bash
git add README.md docs/clean-branch-verification.md
git commit -m "Document practical corruption resume boundary"
```

### Task 3: Final acceptance

**Files:**
- Verify only; no planned modifications

- [ ] **Step 1: Run focused acceptance groups**

Run the corruption, pipeline, reporting, detector-parity, runtime, image-provenance, and
repository-surface groups. Every command must exit zero with no unexpected skip.

- [ ] **Step 2: Run the complete retained suite from the final documentation HEAD**

```bash
/home/yuchen/miniconda3/envs/UE/bin/python -m pytest tests/differential_uncertainty -q -rs
git diff --check
git status --short
```

Expected: the complete suite passes, the diff check is silent, and the worktree is clean.

- [ ] **Step 3: Review the branch diff**

Review `3f0775dbe6ee8c8ea72e9f09646f2a0ef193457b..HEAD`. Confirm the branch remains pretrained
inference-only, retains only the relative-gap differential method, keeps the dedicated report
bundle, exposes the small corruption interface, and makes no claim of strict arbitrary-Python
plugin identity.
