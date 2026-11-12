# Frozen own-R0 archive rescoring control

Pre-result scope: evaluate the two immutable **our-source-trained R0** archives with two fixed feature operators inside the new locked Python3.10/Torch2.7.1+cu128 environment. This is metric-only; no model or checkpoint loading, waveform generation, sample selection or training.

For both full A/train68445 and B/test29421 archives, verify their frozen archive hashes, labels/counts/reference integrity records before computing:

1. `tsfid-source-1000`: unchanged released rFFT-half cutoff and requested256/capped250bins with four channel groups.
2. `tsfid-paper-v1-1024`: the predeclared full501rFFT coefficients pooled to256bins and four groups, no optional truncation.

Compute overall and all six class values for all four archive/operator combinations; no favorable-result stopping or switching settings. Run the independent symmetric covariance audit for each value with the frozen tolerance0.001+0.00001*abs(source-formula score). Archive outputs/config/source/lock hashes under a new name; do not overwrite any original metric or verification sentinel.

Retain old-environment source1000 values beside new-environment source1000 values to quantify evaluation-runtime differences. Compare source1000 versus paper1024 within the new environment to isolate feature-operator effects on the same waveforms. Neither comparison establishes effects of training-time package versions or repairs the original training/model topology. The new1024values remain labeled rescores, not a new generation experiment or established publication lineage.

Use CPU scoring with bounded threads and streaming feature moments, preserving full rows and original labels. The dimensional covariance calculations may be expensive but must not cause sampling or class omission. Keep EEG arrays remote. Freeze the rescoring script, this protocol, all four operator/population definitions and environment lock before execution. Lead must review the concrete script and provenance guards first. This control does not authorize Loop, author checkpoint use or full aligned training.
