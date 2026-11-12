# Research analysis

**Current instruction:** the user's later resumption permits the frozen five-arm source-R0 Loop pilot alongside continued paper alignment; author checkpoints remain prohibited. The pilot is now complete: validation baseline365.6763, loop2 367.2418, loop3 386.4304, flow67 365.2480, flow83 365.0822. Both Loop arms fail the frozen continuation filter. These 2,020-row validation results are distinct from full-train A287.4627 and full-test B365.1448. Astra is reviewing quality/mechanism/compute evidence before further experiments. A separate own-initialized `jet-paper-aligned-v1` target is now running its own-fresh full200epoch seed0 training (PID227768), with no completed new-model score, with explicit paper/source contradictions and retained defaults documented in `PAPER_ALIGNED_V1_PROTOCOL.md`. The missed official checkpoint link was disclosed; its transfer was stopped and no author tensor was loaded, inspected, evaluated or used for initialization. Earlier blanket holds below are historical.

## R0: faithful released execution did not recover the published number

Observation: the unchanged 200-epoch JiT-B/16 checkpoint produces TS-FID 287.4627 under the released train-split wrapper and 365.1448 on the frozen independent test split. The paper reports 235.86. A is numerically 21.9% above that reference, but the exact publication population, feature geometry and checkpoint/sampler lineage are unresolved. A/B use different populations and class mixtures, so their gap is not an isolated overfitting estimate.

Investigation: complete event extraction, reference hashes/labels/counts, checkpoint tensors and identity, actual training/inference arguments, normalization/units, EMA selection, spectral geometry and independent Fréchet arithmetic were checked. All mandatory integrity and numerical checks passed. Root also reconstructed every stored normalized diagnostic scalar and summary. No concrete implementation or arithmetic mismatch was found. Public primary materials did not resolve the publication lineage. Runtime/version differences and one-seed variability remain possible contributors, but no causal contribution has been established.

Decision: preserve the released model and all event weights. Do not select another split, feature extractor, EMA, guidance, epoch or seed to obtain a more favorable number. Released-source R0 is audited; independent reproduction of the paper's number is not established. Astra's original-mission gate review permits a bounded validation-only R1 after baseline freeze. Publication-comparable expansion remains unavailable. See `BASELINE_REPORT.md` and `R0_AUDIT.md` for the four-part judgment and exact evidence.

## R0: marginal and spatial quality limitations

All generated/reference values and native diagnostic primitives are finite. No all-zero windows, undefined correlations or missing edges were found. These checks establish usable measurements, not adequate EEG realism.

The frozen calibrated macro discrepancies are PSD 3.3132, band power 3.0581, morphology 2.2479, temporal 1.7565 and spatial 5.0400, in primitive-specific train/validation real-vs-real scale units. ARTF has PSD 5.9251 and BCKG spatial 11.5598. Different families have different calibration distributions; the normalized values are operational comparisons, not a shared clinical severity scale. W1 and absolute PSD differences do not reveal which domain has larger power. No post-hoc threshold or causal explanation is inferred from these magnitudes.

The source network processes five temporal tokens independently for each channel, sharing weights and class conditioning. With independent channel noise and fixed class y, the sampler is of the form X_i = g_theta(Z_i, y), with independent Z_i and no channel identifier or cross-channel state exchange. Shared deterministic class-time profiles can still influence raw within-window correlations. Repeating the same channel-separable blocks preserves this conditional independence structure; it cannot create arbitrary joint channel dependence.

The separately registered residual diagnostic removes each domain's class/channel/time mean. Real mean absolute class-mean residual correlations range from 0.1683 to 0.3546 across classes; generated values range from 0.00163 to 0.02516. Macro Pearson MAE is 0.27536 and covariance MAE 642.1078 microvolts squared. All pair counts and near-constant checks pass. This is consistent with the source mechanism, not evidence that our model deviated from it. Finite-N mean subtraction shrinks residual variation; matching class counts controls one aspect but does not confer statistical or clinical validation.

Falsifiable implication for R1: per-channel recurrence could alter marginal spectra, morphology and time behavior, or destabilize the pretrained field. It cannot repair missing joint stochastic dependence by itself. A changed raw correlation score would not demonstrate newly modeled cross-channel dependence. If later work adds channel interaction, the mission requires a channel-interaction control without Loop.

## R0: event multiplicity limits interpretation

The train/validation/test archives contain 68,445/15,487/29,421 event rows but only 8,650/1,028/3,243 distinct normalized waveforms. Many annotations select the same full multichannel window; some identical windows carry multiple labels. The complete provenance audit found no label join error. Copied recording payloads do not cross splits, but patient identity cannot be established from nominal directories. The official weighting remains intact.

Neither event rows nor unique waveform hashes are independent biological samples. Validation GPED has only 18 distinct inputs, validation SPSW 23, and test SPSW 46. Future uncertainty must retain whole observable linked-directory components and event multiplicities, as frozen in `PROTOCOL.md`. Inference repeats and 200 overlapping diagnostic subsets do not substitute for independent training seeds.

## R0: where time and memory went

Training took 35,318 seconds; source A evaluation 10,632 seconds; B evaluation 6,441 seconds. The sequential sum is 52,391 seconds (14 h 33 min 11 s), excluding preprocessing, audits, profile and orchestration overhead. The two evaluations generated 97,866 windows, requiring 9,688,734 sample-level conditional network evaluations at 99 NFE. Full evaluation commands include reference collection, archive compression and metrics, not just GPU sampling.

The synchronized shared-A6000 median is 13.1021 seconds per batch of 64, about 4.885 samples/s. Allocated/reserved PyTorch peak is 2,240.337/2,494 MiB; sampled evaluation-process NVML peak is 2,808 MiB. The full training peak was 13,477 MiB allocated and 15,250 MiB sampled NVML. Other GPU work explains observed wall-time variation, but has not been shown to cause the metric gap. Leading-matrix FLOPs are 1.490120146944e12 per sample; this is not a complete all-operation count.

## R1: pending controlled validation experiment

No Loop result is available at this R0 freeze. `R1_PROTOCOL.md` will define the bounded pilot before implementation and candidate generation. It must preserve weights, prove K=1 equivalence, use validation-only fixed rows and paired noise, compare K=2/3 to equal-step and extra-flow-step matched-compute baselines, and record hidden-state/endpoint/vector changes across flow time. Pilot screening is descriptive; it does not establish robust superiority or satisfy any test or expansion gate.

### Engineering update after R0 freeze

The frozen R1 fixtures and K=1 gate are complete. All ten forward comparisons, 99 actual solver evaluations and final-waveform comparisons match the official path with exactly zero measured error; parameter/EMA keys/hashes and checkpoint are unchanged. Astra accepted the engineering evidence. This checks implementation equivalence; it does not resolve the paper's baseline gap or show benefit from recurrence. The five-arm candidate runner remains under review and no candidate result is yet available.

Selected validation support is thin: rows 119/291/512/86/500/512 contain only 23/18/141/50/69/361 unique within-class waveforms and 7/2/3/5/11/38 observable linked components. In particular, GPED represents only two linked components. These fixed exploratory data can expose large intervention failures, but cannot establish broad population generalization or independent-seed superiority. Selection is not being revised after observing this support.
