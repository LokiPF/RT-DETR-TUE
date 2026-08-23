## List of Abbreviations
- id: in distribution
- ood: out of distribution
- mst: maximum spanning tree
- AUROC: Area Under the Receiver Operating Characteristic

## Generating Fréchet Means and Calibration

### Fréchet Means:

A new [solver](src/solver/tue_solver.py) and [engine](src/solver/tue_engine.py) have been developed for Fréchet mean generation. As described in the final report for this guided research, the Fréchet mean generation builds an mst for each linear layer in the decoder based on activation strength. Each vertex represents one neural node in the input and output layer and each edge represents the activation strength of said input node. Prim's algorithm is used for mst generation as it is generally faster compared to Kruskal's algorithm for dense graphs. Prim's algorithm has a time-complexity of $O(E + Vlog(V))$. $V$ is the total number of vertices and $E$ is the number of edges where $E \approx \frac{V^2}{4}$ since $E = V_{in} + V_{out}$. The batched implementation of Prim's algorithm can be found in [tue_utils.py](src/misc/tue_utils.py).

#### Generating Fréchet Means:

1. Set-up the config. Take [rtdetrv2_r18vd_120e_coco_tue_frechet_generation.yml](configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_tue_frechet_generation.yml) as an example. It is important to set ```task: tu_estimation```. Also, ```train_calibration_split_path: /path/to/instances_train_calibration_split.json``` should be set. This file can be generated using [split_train_calibration.py](tools/dataset/split_train_calibration.py).
2. Generate fréchet means using: ```python tools/train.py -c /path/to/config.yaml -r /path/to/pretrained_weights/.pth```. Pretrained weights can be downloaded from 

## Benchmarking

### AUROC
To perform the auroc experiments, create a ```.yaml``` config, similar to [exp1_config.yaml](output/auroc/exp1/exp1_config.yaml). The config contains a set of datasets, incl. their name and if they are ood or id. 

Use the following YAML structure when defining datasets:

```yaml
description: "Description"

datasets:
  - dataset:
      name: "Name of Dataset"
      imgs: "path/to/images"
      anns: "path/to/annotations"
      ood: false

  - dataset:
      name: "Name of Dataset"
      imgs: "path/to/images"
      anns: "path/to/annotations"
      ood: false
```


## Software Tools Used During Development:
| Software Tools | Use Case | Scope | Remarks |
|---|---|---|---|
| ChatGPT-5.6 | Debugging, simplifying, improving code, plotting | Code development | No original ideas taken from AI-tools |
| Claude Opus 4.8 | Debugging, simplifying, improving code, plotting | Code development | No original ideas taken from AI-tools |
| VS-Code | Coding | Code development | IDE |
| uv Astral | Package Management | Code development | To my knowledge currently best and fastest package manager out there (can recommend) |
| Ruff Astral | Formatting | Code development | Python linter and code formatter |
| Code Spell Checker | Spell checking in source code | Code development |  |
|  |  |  |  |

A list of packages required for the code to run can be found in [requirements.txt](requirements.txt). This table is complete to the best of my recollection. If any software tools are missing, please let me know.