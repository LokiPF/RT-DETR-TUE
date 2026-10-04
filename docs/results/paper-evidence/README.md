# Evidence for the paper's claims

The tables behind the claims in `docs/paper-storyline.md` (2 October 2026). Every table covers all 5,000 COCO val images under the 96 conditions, except the forward-pass tables, which use 200 images. The scripts are in `scripts/paper/`. They read `runs/coco/` and never write to it.

```
PAPER_CACHE=<scratch dir> python scripts/paper/stored.py            # CPU, about 15 minutes
PAPER_CACHE=<scratch dir> CUDA_VISIBLE_DEVICES= python scripts/paper/forward.py operations   # CPU, about 25 minutes
PAPER_CACHE=<scratch dir> CUDA_VISIBLE_DEVICES= python scripts/paper/forward.py untrained    # CPU, about 30 minutes
PAPER_CACHE=<scratch dir> CUDA_VISIBLE_DEVICES= python scripts/paper/forward.py untrained_key  # after untrained
PAPER_CACHE=<scratch dir> CUDA_VISIBLE_DEVICES= python scripts/paper/forward.py filters --images 50
python scripts/paper/figures.py
```

`stored.py check` first confirms that its reimplementation of the scores matches the package's two-axis score exactly (maximum difference 0; 0.9169 / 0.8584).

## From the stored statistics (`stored.py`)

| Table | Claim | What it holds |
|---|---|---|
| `depth_curve.csv` | C1 | AUROC of each stage's own score: the level, peak share and flattening against the bank mean (stages 1–4) and against similar scenes (stages 1–3), the activation CDF's five stages, and Hashemi et al.'s encoder maps and decoder |
| `key_stability.csv` | C1 | Per family and severity: the share of the 50 neighbours that survive, the key's move against its neighbourhood's radius, and the shift of the image and of its neighbours' mean, in level and peak share, at stages 1–3 |
| `key_categories.csv` | C1 | COCO panoptic labels: how often the neighbours show a label when the image does and when it does not, against the bank's rate (the Jaccard and object-count numbers are in `stored.json`) |
| `spread_ratio.csv` | C3, C4 | Clean spread around the neighbours' mean against around the bank's mean, per stage, on the 500 z-statistics images and on the 5,000 clean val images |
| `effect_sizes.csv` | C3, C8 | Per family, severity and stage: the shift of the channel levels and peak shares, in bank spreads and in spreads around the neighbours |
| `flatten_or_shift.csv` | C5, C8 | Per family, severity and stage: the change of log level, log top-1% mean and peak share; the summed level and top; the shares of channels that flatten, rise or fall by more than 10%; the share of the shift along the all-channels direction (`flatten_uniform_clean_share.json`: that direction's share of the clean variation) |
| `arms_by_family.csv` | C6 | Per family and severity: each arm's AUROC, the larger arm's, their sum's, and the share of images where the flattening arm is the larger |
| `max_vs_sum.csv` | C6 | The larger arm against the sum on all, untouched and held-out images (the paired intervals are in `stored.json`) |
| `twin_key.csv` | C4, C8 | The scores with the clean twin's neighbours, which removes the key's own drift |
| `ladder_by_severity.csv` | C4 | Global level, level, peak share and two-axis by severity, with paired 1,000-draw intervals on the differences |
| `ablation_k.csv`, `ablation_bank.csv` | C4 | k from 1 to 500; bank sizes 100–2,000, three seeded subsets each |
| `linear_vs_neighbours.csv` | C4 | A ridge map from the key (λ = 10) instead of the neighbours: its AUROC from the image's own key and from the clean twin's, and how far its prediction moves under each corruption |
| `second_axis_statistic.csv` | C5 | The second axis with the raw ratio, the log top-1% mean or the top-1% mean in place of the peak share |
| `opposite_moves.csv` | C5 | Per channel at severity 1, for the flattening families: how peaky the channel is on clean images against how often it gets flatter |
| `fixed_threshold.csv` | Results | A threshold set on the 500 clean z-statistics train images: false alarms on the clean val images, and the share of corrupted images flagged, by group and severity |

## Forward passes on the CPU (`forward.py`, 200 val images)

| Table | Claim | What it holds |
|---|---|---|
| `dose_response.csv` | C5, C8 | Pure operations at graded strengths (gain, contrast, saturation and hue inside and outside the training's jitter range; white veil, offsets, blur, noise) and the benchmark's fog split into its gain, its uniform veil and its plasma structure: AUROCs and channel changes |
| `dose_response_check.json` | | The CPU pass against the stored GPU statistics for the clean and fogged images. The worst case over the 200 images and all channels is within 1% for the stage 1–3 levels, and within 2.5% for the key. Single top-1% means of sparse channels differ by up to 6% at stage 3, and by up to 56% at stage 4, which the method does not use. The package keeps PyTorch's default, which lets cuDNN run the GPU's convolutions in TF32 |
| `untrained_backbone.csv`, `.json` | C1 | The same scores from a random-weight backbone (batch norm re-estimated on 200 clean train images), against the trained one on the same images and conditions: clean and the 19 families at severities 1, 3 and 5 |
| `untrained_key_categories.json` | C1 | COCO panoptic label overlap between each image and its 50 neighbours, for both backbones' keys, against random bank images |
| `first_conv_filters.csv` | C8 | The stem's 32 first filters, trained and random: the response to a uniform offset of 0.1 against the image's own spread, and the share of each filter along R = G = B |

## Figures (`figures/`, drafts)

- `flatten_or_shift`: candidate Figure 1.
- `depth_curve`: candidate Figure 3.
- `dose_response`.
- `ablation`.
