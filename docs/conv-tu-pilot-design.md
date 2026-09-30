# Conv-layer Topological Uncertainty: pilot design

Decisions of 30 September 2026. The pilot plan is
`docs/superpowers/plans/2026-09-30-conv-tu-pilot.md`.

## Question

Topological Uncertainty (TU) computed on the backbone's conv layers: does it flag degraded images and track
detection harm better than two things?

- the same kNN on simpler summaries of the same layer;
- the six COCO baselines, on the same images.

## Decisions

1. **Graph.** Take one conv layer.
   - Nodes are its input cells (c, p) and output cells (d, q).
   - Every product inside the conv is one edge, with weight |K_eff[d, c, t]| · |x[c, q + offset(t)]|.
   - The conv pads with zeros, and edges into that padding do not exist.
   - K_eff is the kernel times the per-output-channel scale |γ| / √(σ² + ε) of the conv's own norm layer. The edges are then the terms of the normalized pre-activation.
2. **Fingerprint.** The K largest values of the graph's 0-dimensional persistence diagram, meaning Kruskal's merge weights.
   - K = 1% of the graph's nodes. The user chose a fraction "so it is also applicable for other detectors".
   - Graphs under 10,000 nodes keep the whole diagram (nodes − 1 values). The decoder's 336-node score-head graph is an example.
3. **Exactness.** The K values are exact. Kruskal reads edges from the heaviest down, and whether an edge joins two groups depends only on heavier edges. So any cut below the K-th value gives the same K values. This is Idea 1 of the page "Conv Graph Reductions".
4. **The cut comes from clean training images** (user: "we should get those 1% from the clean training image").
   - The cut is set per layer from clean COCO train images.
   - An image that needs more edges gets a lower cut automatically.
   - An image with too many edges above the cut gets a higher one.
   - Either way, every image's K values stay exact.
5. **Input size.** RT-DETRv2 fixes it at 640 × 640, in three places:
   - the validation transform (`Resize [640, 640]`);
   - `eval_spatial_size`, which caches the positional embedding and the 8,400 anchors;
   - the TensorRT profile.

   Every image therefore gives the same graph size.
6. **Layers.** One stride-1 3 × 3 conv per backbone stage: `backbone.res_layers[s].blocks[1].branch2a.conv` for s = 0 … 3, the first conv of each stage's second block.

   | Stage | Input (C, H, W) | Nodes | K |
   |---|---|---|---|
   | 1 | (64, 160, 160) | 3,276,800 | 32,768 |
   | 2 | (128, 80, 80) | 1,638,400 | 16,384 |
   | 3 | (256, 40, 40) | 819,200 | 8,192 |
   | 4 | (512, 20, 20) | 409,600 | 4,096 |

7. **Scoring.**
   - Per layer, the score is the mean Euclidean distance to the 5 nearest rows of a bank of 2,000 clean train fingerprints.
   - The distance is not normalized, because fog, contrast and brightness change the size of the activations and that must count.
   - Per-layer distances are z-scored with 500 other clean train images and summed over the four layers.
8. **Controls.** Same layers, same kNN, same z-scoring:
   - **edges:** the K heaviest edge weights, without the cycle rule. This tests whether the topology adds anything.
   - **acts:** the K largest activations |x|, with no kernel and no graph. Graphs under 10,000 nodes pad with zeros.
   - **means:** the mean |x| of each input channel, which is Idea 2's input.
9. **Evaluation.**
   - Images: the first 200 COCO val2017 images of the seed-44 order, with all 96 conditions.
   - The rest follows the existing protocol:
     - LRP labels at the stored threshold 0.55;
     - the stored per-fold λ of ContrastiveConf;
     - the same metrics and paired bootstrap as `docs/coco-baseline-numbers.md`.
   - The nine baseline rows are recomputed on the same 200 images.
10. **Out of scope:**
    - the collapsed-graph arm (Idea 2 as a method);
    - the encoder-token arm;
    - the decoder-query method, whose scores are not stored on this protocol;
    - the fixed-set-of-connections version, which needs node identity across images and was dropped.

## Evidence before the pilot

Two clean train images went through the real checkpoint on the CPU, with a prototype of the exact top-K code. The prototype matches brute-force Kruskal on small layers.

- **Edges needed for K merges.** About 34,000 edges at stage 1 (0.004% of the layer's 936 M) and about 6,000 at stage 4. The graph never has to be built in full.
- **The top 1% is close to the plain heaviest edges.** For the first image, the mean gap between the top-K fingerprint and the K heaviest edges was:

  | K | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
  |---|---|---|---|---|
  | 1% of nodes | 0.4% | 1.9% | 4.1% | 7.3% |
  | 10% of nodes | 2.8% | 4.2% | 5.6% | 13.3% |
  | 50% of nodes | – | – | 18.7% | 25.0% |

  The cycle rule matters more deeper into the diagram and in later stages. The **edges** control measures whether the difference at K = 1% carries any signal.

## What counts as a positive pilot

- **mst** beats **edges** and **acts** on mean AUROC (common families) and on ρ(Δscore, ΔLRP) within conditions. The bootstrap interval of the difference must exclude 0.
- The combined **mst** score is competitive with the best baseline on the same 200 images.
