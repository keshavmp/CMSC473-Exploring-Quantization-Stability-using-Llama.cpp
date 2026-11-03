# P2 Imatrix domain analysis

## Input coverage
- chat: 112 tensors with imatrix data
- math: 112 tensors with imatrix data
- code: 112 tensors with imatrix data

## Preliminary layer ranking
Top five layers by **heuristic** score: 14, 13, 15, 10, 11.

## Interpretation
The scores rank average within-domain tensor activation energy. This is an exploratory proxy, **not** verified quantization sensitivity. Raw values vary by tensor type, so absolute comparisons should be treated cautiously.

## Validation plan
Compare `predicted_layer_ranking.csv` against independent P1 layer-level perturbation measurements. Use `validate` with a `layer,sensitivity` CSV, where higher sensitivity means greater damage.

## Reproducibility
Same BF16 checkpoint, same dataset splits and seed, and same imatrix generation options must be retained for meaningful comparisons.
