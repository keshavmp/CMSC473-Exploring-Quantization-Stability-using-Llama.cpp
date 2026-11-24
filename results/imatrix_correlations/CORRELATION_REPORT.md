# Imatrix feature correlations



These are exploratory associations, NOT evidence that activation importance causes quantization sensitivity.

The imatrix datasets are calibration domains, distinct from benchmark test sets.



## Cross-domain correlations
Compared matched tensor energy across 3 domain pairs. See `domain_tensor_correlations.csv`.

## Weighted error
Not available. Supply `--weighted-errors` from actual BF16/quantized weights. No values were fabricated.

## P1 layer sensitivity
Not yet available. Run controlled single-layer ablations; correlations will be computed when their CSV is supplied.



## Limitations

Correlations are not causal, small samples may be unstable, and layers/tensors from one model are not independent experimental replicates. No significance or confidence claims are made. Compare consistent task metrics and model checkpoints.
