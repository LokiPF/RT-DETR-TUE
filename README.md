## List of Abbreviations
- id: in distribution
- ood: out of distribution
- AUROC: Area Under the Receiver Operating Characteristic

## Benchmarking

### AUROC
To perform the auroc experiments, create a ```.yaml``` config, similar to [exp1_config.yaml](output/auroc/exp1/exp1_config.yaml). The config contains a set of datasets, inlc. their name and if they are ood or id. 

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