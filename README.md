
## Quick start

<details >
<summary>Setup</summary>

```shell

pip install -r requirements.txt
```

The following is the corresponding `torch` and `torchvision` versions.
`rtdetr` | `torch` | `torchvision`
|---|---|---|
| `-` | `2.4` | `0.19` |
| `-` | `2.2` | `0.17` |
| `-` | `2.1` | `0.16` |
| `-` | `2.0` | `0.15` |

Those rows are the **detector-only** combinations. The scene-uncertainty pipeline needs
`torchvision >= 0.18` (torch `2.3`), because its blur loader passes a tuple-returning
`labels_getter` to `SanitizeBoundingBoxes` and earlier versions accept only a single
tensor. `requirements.txt` declares that floor.

</details>

<details open>
<summary>Fig</summary>

<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/437877e9-1d4f-4d30-85e8-aafacfa0ec56">
</div>

</details>


## Model Zoo

### Base models

| Model | Dataset | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS | config| checkpoint | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
**RT-DETRv2-S** | COCO | 640 | **48.1** <font color=green>(+1.6)</font> | **65.1** | 20 | 217 | [config](./configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth) |
**RT-DETRv2-M**<sup>*<sup> | COCO | 640 | **49.9** <font color=green>(+1.0)</font> | **67.5** | 31 | 161 | [config](./configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth)
**RT-DETRv2-M** | COCO | 640 | **51.9** <font color=green>(+0.6)</font> | **69.9** | 36 | 145 | [config](./configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth)
**RT-DETRv2-L** | COCO | 640 | **53.4** <font color=green>(+0.3)</font> | **71.6** | 42 | 108 | [config](./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth)
**RT-DETRv2-X** | COCO | 640 | 54.3 | **72.8** <font color=green>(+0.1)</font> | 76 | 74 | [config](./configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth)
<!-- rtdetrv2_hgnetv2_l | COCO | 640 | 52.9 | 71.5 | 32 | 114 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_l_6x_coco_from_paddle.pth) 
rtdetrv2_hgnetv2_x | COCO | 640 | 54.7 | 72.9 | 67 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_x_6x_coco_from_paddle.pth) 
rtdetrv2_hgnetv2_h | COCO | 640 | 56.3 | 74.8 | 123 | 40 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_h_6x_coco_from_paddle.pth) 
rtdetrv2_18vd | COCO+Objects365 | 640 | 49.0 | 66.5 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_5x_coco_objects365_from_paddle.pth)
rtdetrv2_r50vd | COCO+Objects365 | 640 | 55.2 | 73.4 | 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_2x_coco_objects365_from_paddle.pth)
rtdetrv2_r101vd | COCO+Objects365 | 640 | 56.2 | 74.5 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_2x_coco_objects365_from_paddle.pth)
 -->

**Notes:**
- `AP` is evaluated on *MSCOCO val2017* dataset.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT>=8.5.1$.
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.



### Models of discrete sampling

| Model | Sampling Method | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | config| checkpoint 
| :---: | :---: | :---: | :---: | :---: | :---: |
**RT-DETRv2-S_dsp** | discrete_sampling | 47.4 | 64.8 <font color=red>(-0.1)</font> | [config](./configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp_3x_coco.pth)
**RT-DETRv2-M**<sup>*</sup>**_dsp** | discrete_sampling | 49.2 | 67.1 <font color=red>(-0.4)</font> | [config](./configs/rtdetrv2/rtdetrv2_r34vd_dsp_1x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rrtdetrv2_r34vd_dsp_1x_coco.pth)
**RT-DETRv2-M_dsp** | discrete_sampling | 51.4 | 69.7 <font color=red>(-0.2)</font> | [config](./configs/rtdetrv2/rtdetrv2_r50vd_m_dsp_3x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_dsp_3x_coco.pth)
**RT-DETRv2-L_dsp** | discrete_sampling | 52.9 | 71.3 <font color=red>(-0.3)</font> |[config](./configs/rtdetrv2/rtdetrv2_r50vd_dsp_1x_coco.yml)| [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_dsp_1x_coco.pth)


<!-- **rtdetrv2_r18vd_dsp1** | discrete_sampling | 21600 | 46.3 | 63.9 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp1_1x_coco.pth) -->

<!-- rtdetrv2_r18vd_dsp1 | discrete_sampling | 21600 | 45.5 | 63.0 | 4.34 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp1_120e_coco.pth) -->
<!-- 4.3 -->

**Notes:**
- The impact on inference speed is related to specific device and software.
- `*_dsp*` is the model inherit `*_sp*` model's knowledge and adapt to `discrete_sampling` strategy. **You can use TensorRT 8.4 (or even older versions) to inference for these models**
<!-- - `grid_sampling` use `grid_sample` to sample attention map, `discrete_sampling` use `index_select` method to sample attention map.  -->


### Ablation on sampling points

<!-- Flexible samping strategy in cross attenstion layer for devices that do **not** optimize (or not support) `grid_sampling` well. You can choose models based on specific scenarios and the trade-off between speed and accuracy. -->

| Model | Sampling Method | #Points | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | checkpoint 
| :---: | :---: | :---: | :---: | :---: | :---: |
**rtdetrv2_r18vd_sp1** | grid_sampling | 21,600 | 47.3 | 64.3 <font color=red>(-0.6) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp1_120e_coco.pth)
**rtdetrv2_r18vd_sp2** | grid_sampling | 43,200 | 47.7 | 64.7 <font color=red>(-0.2) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp2_120e_coco.pth)
**rtdetrv2_r18vd_sp3** | grid_sampling | 64,800 | 47.8 | 64.8 <font color=red>(-0.1) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp3_120e_coco.pth)
rtdetrv2_r18vd(_sp4)| grid_sampling | 86,400 | 47.9 | 64.9 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_120e_coco.pth) 

**Notes:**
- The impact on inference speed is related to specific device and software.
- `#points` the total number of sampling points in decoder for per image inference.


## Usage
<details>
<summary> details </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config --use-amp --seed=0 &> log.txt 2>&1 &
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -r path/to/checkpoint --test-only
```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -t path/to/checkpoint --use-amp --seed=0 &> log.txt 2>&1 &
```

<!-- <summary>4. Export onnx </summary> -->
4. Export onnx
```shell
python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check
```

<!-- <summary>5. Inference </summary> -->
5. Inference

Support torch, onnxruntime, tensorrt and openvino, see details in *references/deploy*
```shell
python references/deploy/rtdetrv2_onnx.py --onnx-file=model.onnx --im-file=xxxx
python references/deploy/rtdetrv2_tensorrt.py --trt-file=model.trt --im-file=xxxx
python references/deploy/rtdetrv2_torch.py -c path/to/config -r path/to/checkpoint --im-file=xxx --device=cuda:0
```
</details>



## Citation
If you use `RTDETR` or `RTDETRv2` in your work, please use the following BibTeX entries:

<details>
<summary> bibtex </summary>

```latex
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140}, 
}
```
</details>
# RT-DETR-TUE

## Scene uncertainty pilot

Class-independent persistence scene uncertainty: a scene is scored by how far its decoder
queries sit from a bank of queries drawn from clean reference images, and the study asks
whether that score rises as the scene is blurred. One entry point, seven subcommands, run in
order -- six that build the artifacts, and one cache-only analysis over what they wrote.

**A raw kNN score is not a corruption probability.** It is a mean distance to the *k*
nearest bank vectors, divided by the inter-quartile spread of the same distance measured on
clean reference queries, and averaged over the three decoder layers. Nothing in this
pipeline is fitted against a corruption label, so a score is comparable across the
severities of one configuration and means nothing on its own scale. `report` publishes
trend statistics, not calibrated probabilities.

```bash
UE_PY=/path/to/python
COCO=dataset/coco
CKPT=pretrained_weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
CFG=configs/scene_uncertainty/rtdetrv2_r18vd_coco.yml
OUT=output/scene_uncertainty/pilot

# Record the commit in every artifact. `git rev-parse` fails on a host running from an
# rsync'd tree -- provenance then degrades to "unknown" with a warning -- so take the sha
# from wherever the code was edited and pass it in. The value is compared when an
# interrupted extraction is resumed: export it for every command of a run, or for none.
export SCENE_UNCERTAINTY_GIT_COMMIT=9184adb65864b802378ad3e261b7ea62be5a3b98  # <- your sha

$UE_PY tools/scene_uncertainty.py select \
  --train-ann $COCO/annotations/instances_train2017.json \
  --val-ann $COCO/annotations/instances_val2017.json \
  --output $OUT/splits

$UE_PY tools/scene_uncertainty.py extract-reference \
  --config $CFG --checkpoint $CKPT \
  --images $COCO/train2017 \
  --annotations $COCO/annotations/instances_train2017.json \
  --selection $OUT/splits/reference.json \
  --output $OUT/reference_cache

$UE_PY tools/scene_uncertainty.py extract-blur \
  --config $CFG --checkpoint $CKPT \
  --images $COCO/val2017 \
  --annotations $COCO/annotations/instances_val2017.json \
  --selection $OUT/splits/evaluation.json \
  --output $OUT/blur_cache

$UE_PY tools/scene_uncertainty.py build-bank \
  --cache $OUT/reference_cache \
  --population coverage --capacity 25000 \
  --output $OUT/bank_coverage_25k

$UE_PY tools/scene_uncertainty.py evaluate-knn \
  --cache $OUT/blur_cache --bank $OUT/bank_coverage_25k \
  --normalization raw --k 5 --partition tuning \
  --output $OUT/results/raw_k5.csv

$UE_PY tools/scene_uncertainty.py report \
  --results $OUT/results/raw_k5.csv \
  --output $OUT/reports/raw_k5
```

Add `--limit 4 --batch-size 1 --num-workers 0` to both extract commands for a four-image
smoke run in a scratch `--output`; every later command works unchanged on the smaller
artifacts (use a smaller `--capacity`).

The default `--partition tuning` scores the 250 tuning images and leaves the 250 test
images cached but unscored. Spend the held-out half once, after the policy and
normalization have been chosen on the tuning half -- the names below are a placeholder for
whatever that choice turns out to be, not a recommendation:

```bash
$UE_PY tools/scene_uncertainty.py evaluate-knn \
  --cache $OUT/blur_cache --bank $OUT/bank_coverage_25k \
  --normalization raw --k 5 --partition test \
  --policies top20,smooth_1 --aggregations mean,median \
  --output $OUT/results/final_raw_k5.csv
$UE_PY tools/scene_uncertainty.py report \
  --results $OUT/results/final_raw_k5.csv \
  --output $OUT/reports/final_raw_k5
```

### Confidence deciles over the artifacts that already exist

Which queries carry a scene's trend? `analyze-confidence-deciles` splits each image's queries
into ten bins by the detector's own confidence and scores every bin twice -- once with the
persistence distance and once with `1 - confidence`, over the same queries and through the
same scene summary -- so the two signals are compared as trends and never as magnitudes. The
bins are built both `dynamic` (re-ranked at every severity) and `frozen` (severity zero's
membership reused), which separates queries changing bin from fingerprints moving.

```bash
$UE_PY tools/scene_uncertainty.py analyze-confidence-deciles \
  --cache $OUT/blur_cache \
  --results $OUT/results/raw_k5.csv \
  --output $OUT/reports/confidence_deciles_raw_k5
```

**Nothing is recomputed.** `--cache` supplies the cached logits, boxes and per-layer
persistence fingerprints; `--results` supplies the saved per-query distances, the fitted layer
score scales and the run provenance, read from the `.query_distances.pt`, `.normalizers.pt`
and `.manifest.json` siblings named beside that CSV. There is no detector forward pass, no
bank build, no kNN search, no training and no re-fitting of a normalizer or any other
calibration: every score is a saved distance divided by the scale the result artifact already
carries. So the command costs minutes on a CPU rather than GPU hours, and it can be rerun
against a different result set over the same cache. (Confidence itself is recomputed from the
cached logits as each query's largest sigmoid class score, and is deliberately *not* the
cache's own `confidence` field, which was reduced before the float16 cast and is wrong by
~1e-5 -- invisible to a threshold, decisive for a rank.)

**Tuning results only.** There is deliberately no `--partition` flag. A result manifest whose
`source_partition` is `test` or `all` is refused, as is any individual saved-distance row
labelled that way, so the held-out half cannot be spent from this command line. The blur cache
holds both partitions by design, so *its* test records are skipped rather than refused -- the
gate is on the result artifact, not on the cache. The join is proved before anything is scored:
both manifests by content address, then key, layer set and query count for every record. An
image missing any of the six severities, or left with fewer than ten non-padded queries, stops
the run rather than being scored in part.

`--output` is refused when it already holds a finished report, which means a directory
containing `summary.json`; nothing is ever overwritten. A directory left behind by a run that
died part-way *is* written into, because there is nothing in it worth preserving: the seven
artifacts below are published together or not at all, and a run that fails leaves at most an
empty directory, never a readable report of six files. (The seven publications are seven
renames rather than one transaction, so a process *killed* inside that loop can still leave a
prefix; what is ruled out is a failure of the analysis itself.)

* **`per_scene.csv`** -- every scored row, keyed by `image_id`, `severity`, `signal`,
  `score_scope`, `confidence_bin`, `membership_mode`, `padding_mode` and `aggregation`, with
  `score`, `selected_count` and `clean_overlap` beside it. The per-row `selected_query_ids`
  lists are held in memory and deliberately kept out of the file.
* **`summary.json`** -- the grouped metrics; each persistence group beside its matched
  confidence control; each dynamic bin against its frozen twin; the lowest bin with and without
  the padding filter; every candidate against the published all-query benchmark; and a ranking
  restricted to the deployable ones -- persistence only, primary scope, padding-filtered,
  non-frozen membership, full severity coverage. Paired image counts sit next to every rate,
  and wins, ties and losses are reported separately, because a per-image Spearman over six
  severities takes only 35 distinct values and exact ties are common.
* **`easy-report.md`** -- the same summary in prose, generated from that summary and nothing
  else, so every sentence in it can be checked against `summary.json`.
* **`blur_curves.png`** -- median clean-relative score against severity, per decile bin, with
  each image's own severity-0 score subtracted so the curve shows the rise rather than the
  level. Persistence and confidence get separate panels because the two share no unit.
* **`confidence_decile_heatmap.png`** -- median Spearman by confidence bin and signal, always
  all ten bins, with an empty bin drawn as `n/a` rather than omitted.
* **`dynamic_vs_frozen.png`** -- each bin's dynamic result beside its frozen twin, side by side
  and never averaged into one number.
* **`padding_sensitivity.png`** -- the lowest bin repeated with the padded-query union removed
  and kept.

As with `report`, nothing here is fitted against a corruption label. Every statistic published
about a score is invariant under a positive affine rescale of it, which is the only sense in
which a persistence distance and `1 - confidence` can be compared at all.

### Reading the artifacts

* **One writer per `--output`.** The immutability guard is a file check, so two extractions
  aimed at one directory both pass it and silently overwrite each other's shards. Run the
  commands in sequence.
* **`near_duplicate_fraction`, in `<results>.manifest.json` under `clean_distance_fit`.**
  The share of sampled bank rows whose nearest other row is closer than half the typical
  clean distance. Every score is divided by the `scale` fitted beside it, and near-duplicate
  rows inflate that `scale` -- 20% twins inflated it about 2.8x on a synthetic bank. Above 1%
  the run says so on stderr, and **the commands above do print that warning for all three
  layers**: the 25,000-vector coverage bank in this repository reads 0.023-0.034 (0.043-0.047
  at 50,000), because the detector emits bit-identical outputs for several query slots at
  once, so ~1.6% of bank rows are exact duplicates of another row. That is expected here, not
  a misconfiguration. What an inflated `scale` changes is the *amplitude* of every curve --
  the score axis is compressed -- and not the rank statistics the report leads with
  (`median_spearman`, `mean_adjacent_monotonicity`), which a positive rescale cannot move.
* **`scored_image_count` and `scored_severity_count`, in `summary.json`.** A policy that
  stops selecting queries under blur produces no score at those severities, and the trend
  statistics are computed over what survived. A curve truncated at severity 2 and a curve
  that rose across the whole sweep both publish `median_spearman: 1.0`; the counts beside
  them, and `empty_selection_frequency`, are the only things that tell them apart.
* **`has_class_switch` is blind to detection loss.** It compares only annotations matched at
  both severities, so `no_switch` means "nothing that was still detected changed class", not
  "the detector was unaffected". Blur mostly makes detections disappear, and that does not
  appear in this split at all.
* **`smooth_*` policies are reported under `weighted_mean`.** A smoothed policy carries its
  signal in per-query weights, so its aggregation is forced to the weighted mean and the row
  is labelled with the aggregation that actually ran, whatever `--aggregations` asked for.

### What the first bounded pilot measured

Run on `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` over 5,000 reference and 500 evaluation
images (250 of them scored, the other 250 held out): 50 min to extract the reference cache
and 30 min to extract the six-severity blur cache on one RTX 4090 (156 MiB peak CUDA), then
under two minutes for the bank, the 69,000 scored rows and the report.

**The raw score fell as the blur got worse for every confidence-selected policy.**
`top20/mean` reached a median Spearman of -0.89 across the 250 tuning images, and the whole
top-K, threshold and smoothed family sat between -0.6 and -0.9. Only the upper-quantile
aggregation over all queries rose at all (`all/q90`, +0.43), and it dips before it rises.
Most of that drop is not about *which* queries get selected. Freezing each image's
severity-0 top-20 query ids and re-scoring that same frozen set at every severity still gives
a median per-image Spearman of **-0.771** (medians +0.175, +0.142, +0.091, +0.039, -0.107,
-0.161): the surviving queries themselves move *toward* the reference bank as the image is
blurred, because the persistence vector shrinks toward the origin, into the dense
low-magnitude region of a bank that is 54% background. That frozen set is 71% of the drop by
amplitude (0.336 of the 0.471-IQR span) and 87% of it by median Spearman; selection churn
supplies only the remaining 13-29%, with median Jaccard against the severity-0 selection
collapsed to 0.29 by severity 1.

Read `summary.json` before assuming a rising curve. The trend rides on persistence-vector
*magnitude*, and two of the four `--normalization` modes keep that magnitude: `raw`, and
`robust_z`, which is a per-feature affine rescale rather than a magnitude removal. `unit`
discards it and `shape_scale` demotes it to one coordinate in 336. `robust_z` is the control
that separates per-dimension rescaling from magnitude removal, and none of the three
alternatives has been run yet.
