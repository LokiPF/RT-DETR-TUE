# Practical corruption plugin design

**Status:** Approved on 2026-08-25

## Purpose

Keep the corruption extension point easy to use. A corruption only needs to produce the
requested transformed image. The workflow will not attempt to prove the complete identity of
arbitrary Python functions, imported objects, globals, or in-memory library state.

This addendum supersedes the strict plugin-implementation identity requirements added during
the final provenance review. It does not weaken the existing checks for input images,
manifests, checkpoint bytes, detector/runtime settings, cached artifact contents, or atomic
publication.

## Corruption interface

The existing small interface remains the complete contract:

```python
class Corruption(Protocol):
    name: str
    severities: tuple[Severity, ...]

    def apply(self, image: Image.Image, level: int) -> Image.Image:
        ...
```

`Severity` contains the ordered integer level and its reportable parameter. Gaussian blur
continues to provide levels 0 through 5 with radii 0, 1, 2, 4, 8, and 12. A future corruption
implements the same three members and needs no implementation SHA, behavior-state mapping,
dependency mapping, source inspection, or special adapter.

The run snapshots the corruption name, severity table, and bound `apply` operation once at the
start. It validates the name and ordered severity table, but it does not recursively inspect or
revalidate Python internals.

## Output-folder and resume contract

One output folder belongs to one fixed experiment. Resume is intended only for continuing an
interrupted run with the same corruption code and settings.

The recorded provenance contains the corruption name and ordered severity table. A different
name or severity table is refused in the same output folder. A code-only change that preserves
those values is not automatically detected. After changing a corruption implementation,
imported helper, library behavior, or hidden setting, the user must choose a new output folder.

This is an explicit practical boundary, not a claim that arbitrary Python behavior can be
fingerprinted. The README and generated report must state this boundary plainly.

## Retained safety and reproducibility checks

The simplification is limited to plugin introspection. The workflow continues to bind and
verify:

- image content and stable file identity;
- reference/evaluation manifest contents and non-overlap;
- checkpoint bytes;
- fixed scientific configuration;
- resolved device, batch size, shard size, software versions, and applicable GPU details;
- extraction roster, shard hashes, artifact hashes, and output-directory ownership; and
- final image and artifact consistency before successful publication.

The final audit no longer calls plugin-identity inspection. Its last substantive checks remain
the full image audit, artifact sweep, and final lightweight image signatures.

## Reporting

Provenance, summary JSON, and Markdown report show the corruption name and severity mapping.
They do not show or promise complete module, callable, build, state, or dependency hashes.

The numerical method is unchanged: the same pretrained detector path, padding union, dynamic
confidence groups, relative-gap scores, AUROC, Spearman correlation, curve checks, bootstrap,
figures, and report bundle remain in place.

## Failure behavior

The run fails before cache reuse when the recorded corruption name or severity table differs.
Invalid severity tables and invalid `apply` results still fail clearly. The workflow does not
try to detect code-only or in-memory changes hidden behind an unchanged name and severity table;
the documented remedy is a new output folder.

## Tests

The retained tests must demonstrate:

- Gaussian blur produces the expected identity and blurred images at its fixed levels;
- a small second corruption works through the unchanged end-to-end pipeline;
- an interrupted run resumes with the same corruption without repeating completed inference;
- a changed corruption name or severity table is rejected in an existing output folder;
- reports accurately record the corruption name and severity mapping;
- strict implementation/dependency/global-state inspection is absent; and
- all retained input, checkpoint, runtime, cache-integrity, numerical, reporting, detector-parity,
  and repository-surface tests remain green.

## Acceptance

The change is complete when the public corruption interface is again only `name`, `severities`,
and `apply`; the strict plugin-introspection code and its dedicated tests/documentation are
removed; the complete retained suite passes; and the verification record reflects the tested
code commit and current repository counts.
