# Scene-reliability query explainer design

**Date:** 2026-08-26  
**Status:** Approved for planning

## Purpose

Create a Markdown reference that gives construction and agricultural vehicle design
teams, with no computer-vision or machine-learning background, an intuitive but
accurate understanding of the query-based scene-reliability method implemented in
this repository. The reference helps them recognise what the score says about
operating-condition limitations; it does not prescribe a downstream company policy.

## Audience and tone

- Vehicle design and systems teams, not machine operators.
- No assumed knowledge of computer vision, neural networks, transformers, or
  query-based detection.
- A durable reference rather than a short onboarding note.
- Plain language first. Use short definitions, concrete construction and agriculture
  examples, and lightweight equations only after their meaning is established.
- Describe queries as candidate object slots that gather image evidence and
  coordinate with one another. Do not explain transformer-layer internals.

## Narrative approach

Use a scene-reliability-first structure. Begin with conditions in which an otherwise
capable perception system may be operating outside familiar visual conditions:
heavy rain, fog, dust, glare, darkness, mud- or water-contaminated lenses, and
terrain or scene appearance changes. Treat construction and agriculture equally.

Explain the detector only when needed to establish the uncertainty method:

1. A camera scene produces many queries, each a candidate object hypothesis with
   an internal behaviour pattern.
2. A compact topological fingerprint records the connection pattern of that
   behaviour. It is a behavioural signature, not an object label or a probability.
3. A bank of fingerprints from clean, familiar scenes provides a comparison ruler.
4. For a new scene, each selected query receives an unfamiliarity distance from its
   nearest fingerprints in the bank.
5. A single scene score compares two confidence-ranked query groups rather than
   relying on an absolute distance alone.

## The two-group scene score

Give the group-selection rationale its own section, titled along the lines of
“How we found a useful scene ruler.” Explain that controlled experiments apply
increasing image degradation to otherwise identical scenes, compare candidate
query-group choices, and retain choices that make degradation easier to rank and
separate from clean scenes.

The deployed policy uses:

- **Ruler group:** the highest-confidence valid queries (90–100% confidence
  decile). Their mean clean-bank distance acts as an in-scene reference.
- **Changing group:** middle-confidence valid queries (50–60% decile). Their mean
  distance was more responsive to worsening image conditions.
- **Scene score:** the relative difference between these two mean distances,
  `2 × (changing − ruler) / (changing + ruler)`.

Interpret the result conversationally before showing the formula: a larger gap means
the changing part of the query population is behaving more unusually than the
scene’s stable high-confidence reference group. State clearly that this choice is an
empirical result for this method and data—not a universal rule for all query-based
detectors.

## Proposed reader-facing structure

1. Why scene reliability matters in off-road environments
2. The question this method answers
3. Minimum background: camera evidence and coordinated queries
4. Fingerprints of internal behaviour
5. Building a bank of familiar clean-scene behaviour
6. Measuring unfamiliarity in one new scene
7. How we found a useful scene ruler
8. Turning two query groups into the final score
9. Reading the score responsibly: what it supports and what still requires
   validation and company-specific policy
10. Parallel worked examples for a dusty construction scene and a foggy/rainy
    agricultural scene
11. Plain-language glossary

## Visuals

Use Markdown-native diagrams (ASCII flow diagrams and, where rendered, Mermaid) so
the file remains standalone. Include:

- scene → queries → fingerprints → clean bank → group comparison → scene score;
- a side-by-side ruler-group / changing-group illustration;
- a compact worked-example comparison for construction and agriculture.

## Evidence package and placement

Place the final reader-facing reference and its evidence files directly in `docs/`:

- `scene-reliability-reference.md` — the reference itself, including a compact reader-facing results table;
- `scene-reliability-group-reaction.png` — a publication-ready line chart of the recorded confidence-group comparison results across the five non-clean Gaussian blur levels;
- `scene-reliability-group-evidence.csv` — the exact plotted values and labels.

The figure and table will reproduce the recorded tuning-study comparison between:

- the planned 0–10% ruler versus 50–60% responsive relative-gap comparison; and
- the later 90–100% versus 50–60% raw-gap comparison, which was suggested after inspecting the earlier tuning results.


The chart will show AUROC by blur level (1–5), label the 250-image Gaussian-blur tuning set, and make clear that the later comparison is exploratory. It explains why group behaviour was investigated; it is not a performance claim for the current fixed 90–100% / 50–60% relative-gap workflow.


## Accuracy boundaries

- The implementation is inference-only; it does not retrain the detector.
- The score measures unfamiliarity of internal detector behaviour relative to a
  clean reference bank. It is not a probability that a detection is wrong and does
  not itself measure object-detection accuracy.
- It does not dictate whether a product logs, alerts, limits automation, or takes
  any other downstream action.
- Current repository results use Gaussian blur as a controlled degradation. Rain,
  fog, and dust are motivating deployment examples that require representative
  validation before any conclusion about performance under those conditions.
- The included group-reaction chart reproduces recorded tuning-study numbers
  and retains the planned-versus-exploratory provenance of the two comparisons.

- Avoid unnecessary implementation detail such as transformer-layer mechanics,
  exact tensor shapes, or full topological-persistence derivations.

## Verification

Before delivery, check that the Markdown renders cleanly, the narrative explains all
introduced terms before relying on them, both domains receive balanced examples, the
formula agrees with the implemented score, and every limitation above remains clear.
