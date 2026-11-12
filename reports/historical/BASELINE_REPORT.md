# R0 baseline reproduction

**Current status: our fresh paper-aligned200epoch training is running; author checkpoints remain prohibited.** The completed200epoch seed0 released-source run produced TS-FID287.4627 on A and365.1448 on B; integrity and numerical checks passed. These facts do not reproduce the paper's235.86. The later user-authorized frozen source-R0 five-arm validation pilot is complete, with both Loops failing the continuation filter (R1_REVIEW.md). The separate `jet-paper-aligned-v1` full run follows the explicit main-method interpretation in `PAPER_ALIGNED_V1_PROTOCOL.md`, with unresolved author details declared. At2026-11-12 04:23:58UTC, childPID227768 completed epoch0/199 all267steps with finite meanloss1.6161790. Full training and final scoring are incomplete. See ACTIVE_RUNS.md for exact hashes, owned PIDs and logs.

## Publication and implementation

**Metric populations must remain explicit.** The original own source checkpoint scores287.4627 on all68,445 training rows and365.1448 on all29,421 test rows. The later R1 baseline365.6763 uses a fixed2,020-row validation subset with different class proportions; it is not a deterioration of the training score. R1's five arms share identical references and initial noise. All five arms are now complete, with no Loop arm passing the frozen continuation filter. Paper-aligned fresh training is now running; see ACTIVE_RUNS.md for the current evidence and phase.

The proceedings paper reports TUEV TS-FID 235.86, Silhouette 0.983 and downstream accuracy gain +0.032 in Table 1. Appendix D.1 repeats 235.86 for Gaussian initialization and loss weights (1, 1, 0.1, 0.1), and describes the scores as held-out. [ICML proceedings paper](https://proceedings.mlr.press/v306/wang26fp.html)

The clean detached server checkout and local official repository both resolve to `07f9e6491796f4f2c717d6259b1a2e24afce6a77`. The initial checkpoint search covered the tracked tree, GitHub releases and project page but missed the official README's Hugging Face link. A fresh run was trained on the basis of that incomplete search. During renewed alignment, the public model API revealed an official TUEV weight at revision `8463adb7dbdff0ba26f4d9b50292abf2f7e1b31d`. Root disclosed the checking omission. The user then explicitly prohibited using that checkpoint and requested aligning our own reproduction. The transfer was stopped; no tensor loading, inspection, evaluation or initialization used author weights. The completed source run and subsequent paper-aligned version use our own initialization only. [Pinned repository](https://github.com/Y-Research-SBU/JET/tree/07f9e6491796f4f2c717d6259b1a2e24afce6a77)

| Question | Evidence |
| --- | --- |
| What architecture is implemented? | 129,865,160 total parameters verified from the pinned CPU model, including fixed sinusoidal position parameters. Twelve 768-wide blocks with 12 heads operate on five temporal tokens per channel. Channels share weights and conditioning. Each channel's attention and MLP activations are separate. |
| What does the official wrapper optimize? | Endpoint L1, patchwise statistics and total variation, and per-example flattened Pearson loss. Optional STFT weight is zero. Inverse-frequency weighted sampling, BF16 autocast, AdamW, and dual EMA are in the released trainer. |
| Which reference samples are used? | Public wrapper: full training split with matched class counts. The upstream selector uses a seeded permutation without replacement; frozen full-split indices cover each processed file once. The completed source-A archive passed this verification. CLI default: `test`. Published Table 1 split and its mapping to the Appendix D.1 held-out set remain unconfirmed. |
| What features are used? | The public evaluator requests 256 bins but emits 250 retained FFT coefficients × four channel groups for TUEV. Verified feature shape `[4,1000]` on synthetic input. Appendix A.2 states 256 pooled bins. |
| What sampler is used? | The released command uses EMA1, Gaussian noise, Heun 50 including an Euler final step, CFG 1 and seed 0. The publication leaves the exact settings for the Table 1 result unconfirmed. |
| What secondary metrics are reproducible from the public evaluator? | The CLI emits aggregate and per-class TS-FID. Appendix A.2 gives descriptions of Silhouette and CBraMod augmentation accuracy. Their exact experimental settings and executable implementations still need evidence. |

The code-to-paper gap concerns both the reference population and possibly the feature geometry. Closeness to 235.86 on a chosen split alone cannot establish lineage. Record all preregistered tracks and report the comparability judgment independently of the observed scores.

The released sampler starts each channel from independent Gaussian noise and applies the same deterministic, channel-separable network at each solver step. Conditional on a class, the architecture therefore generates exchangeable channels without a mechanism for channel-specific stochastic dependence. Shared class-dependent time profiles can still contribute to correlations computed within a window. The distinction matters when interpreting spectral and spatial diagnostics.

## Server and data

The server has an NVIDIA RTX A6000 (49,140 MiB), driver 560.35.03 and Miniconda. The existing raw TUEV v2.0.1 EDF tree occupies 19 GiB. Other GPU work affected throughput during training. Measured training time and process memory are recorded below.

| Official raw split | EDF recordings | Recording-directory IDs | Annotations |
| --- | ---: | ---: | ---: |
| Train pool | 359 | 290 | 83,932 |
| Eval | 159 | 80 | 29,421 |

The raw manifest found zero train/eval recording-directory ID overlap, zero train filename-prefix mismatches against the recording directory, zero basename collisions, and zero missing or invalid REC files. Every raw annotation was one second. Some official eval filenames begin with class labels, so the independent audit uses recording directories for grouping. A common person-identity namespace across the raw pools is unavailable. Astra approved unchanged official preprocessing and training after reviewing the processed manifest and smoke test.

Raw EDF hashes and combined EDF/REC hashes have zero cross-split matches. Some EDF content repeats within the train pool with distinct annotations. The validation audit checks whether identical EDFs or annotated windows cross the released train/validation split. Directory separation and zero cross-split byte matches support the released train/eval separation. Anonymized EDF headers do not establish a common patient namespace across the two pools.

The isolated Miniconda prefix is `~/EEG-Loop/conda-r0`. Its locked runtime uses Python 3.11.5, PyTorch 2.6.0+cu124, MNE 1.13.2, NumPy 2.4.6 and SciPy 1.15.3. The official README recommends Python 3.10 and a cu128 PyTorch wheel. Hardware and runtime are documented reproduction differences.

The unmodified preprocessor completed. All 113,353 annotated events produced finite 16 × 1000 outputs with valid labels. The three processed splits have zero pairwise directory ID overlap. The 20 duplicate EDF groups in the raw train pool each use different directory names within a group. No duplicate EDF group crosses the released train/validation split.

| Processed split | Recording-directory IDs | Total | SPSW | GPED | PLED | EYEM | ARTF | BCKG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 232 | 68,445 | 526 | 10,963 | 5,515 | 984 | 10,553 | 39,904 |
| Val | 58 | 15,487 | 119 | 291 | 669 | 86 | 500 | 13,822 |
| Test | 80 | 29,421 | 567 | 4,677 | 1,998 | 329 | 2,204 | 19,646 |

The processed splits are under `~/EEG-Loop/tuev-official`. Dataset hashes and amplitude summaries are in `~/EEG-Loop/evidence/processed-manifest.json`. Raw audit evidence, duplicate-window checks and environment freeze are under the same evidence directory.

Frozen reference indices contain 68,445, 15,487 and 29,421 distinct processed file paths for train, validation and test respectively. Their distinct normalized float32 waveform hashes number 8,650, 1,028 and 3,243. Some identical waveforms have multiple labels. File counts therefore describe event-row multiplicity rather than independent waveform support; even distinct overlapping windows are not independent observations. The official multiplicities remain intact. Aggregate coverage is recorded in `~/EEG-Loop/evidence/reference-index/coverage-summary.json`. Both completed evaluation archives passed the frozen full-reference integrity verification.

The completed REC provenance join verified all 113,353 annotation labels and 1,553 deterministic pickle channel/label spot checks without mismatches. Repeated rows at the same recording and extraction coordinates account for 59,748 redundant train rows, 14,449 validation rows and 26,136 test rows. Identical whole-file EDF copies account for another 47 train and 10 validation rows. Another 42 redundant test rows occur across files with identical sample payloads and extraction coordinates but different headers. Multilabel waveform groups number 344, 32 and 144. The released preprocessor emits the full 16-channel window for each annotation and retains its class; the loader does not use offending-channel metadata. These results describe that annotation-to-example mapping. Detailed provenance remains private under `~/EEG-Loop/evidence/event-provenance-v2/`; root independently reconciled the exported row, coordinate and aggregate counts.

The stronger recording audit hashed all 518 EDF sample payloads and checked their channel, calibration and sampling metadata. There are 481 distinct payloads and 37 repeated-payload groups, all with matching interpretation metadata. No payload crosses the processed train, validation or test boundaries. This supports exact recording-copy separation while person identity remains unverified. All 274 training/validation linkage components equal the frozen calibration membership sets exactly. The held-out data have 63 observable linkage components across 80 directory IDs. Their private map and future clustered-uncertainty rule were frozen before formal R0 scores; original metric inputs, multiplicities and diagnostic settings remain unchanged. Checksums for eight outputs in `~/EEG-Loop/evidence/all-edf-payload-v1/output.sha256` passed.

The one-epoch engineering smoke completed all 267 steps with mean training loss 1.2498, 0.4015 s/step, and 13,475 MiB peak allocated CUDA memory. The smoke checkpoint reloaded with model, both EMA states, and AdamW state. Its artifacts stay separate from the full run.

After Astra reviewed the integrity and smoke evidence, the fresh seed-0 run started the unchanged 200-epoch official wrapper. Training coordinator PID 21546 was verified active with finite loss at epoch 0, step 40/267. Initial engineering throughput projected roughly six hours before evaluation. Later GPU sharing increased step time from about 0.40 to 0.79 seconds, so the initial duration estimate is obsolete. The run logs to `~/EEG-Loop/evidence/baseline-seed0.log` and writes to `~/EEG-Loop/baseline-seed0`.

After training completed and final checkpoint hashes matched, evaluation coordinator PID 21547 automatically ran the released train protocol and then the independent test protocol in separate output directories. Both commands and all scheduled diagnostics completed successfully. Its log is `~/EEG-Loop/evidence/evaluation-coordinator.log`.

## TS-FID numerical cross-check

`audit_ts_fid.py` uses the released spectral extractor and independently calculates means, covariances and the Fréchet covariance term. It replaces the nonsymmetric covariance product square root with the eigenspectrum of the symmetric matrix formed by sandwiching the generated covariance between square roots of the reference covariance. The numeric acceptance tolerance, frozen before formal evaluation, is 0.001 plus 0.00001 times the absolute released score.

All seven comparisons in the official GPU smoke archive passed. The maximum absolute difference was 0.00008450. The verified feature dimension was 1000. Formal A and B each passed all seven comparisons. A's symmetric overall value was 287.46273162829345, differing from the released result by 0.000000071611; its maximum class difference was 0.00001626735. B's symmetric overall value was 365.1448073854919, differing by 0.000000060225; its maximum class difference was 0.00002750317. Reports are under `~/EEG-Loop/evidence/evaluation-integrity/`. These checks validate the released arithmetic; publication lineage remains a separate question.

## Additional publication materials

Astra inspected the [official ICML poster](https://icml.cc/virtual/2026/poster/63823). It repeats the TUEV 235.86 score but supplies no evaluation split, sample count, frequency pooling configuration or checkpoint. The linked [SlidesLive report](https://slideslive.com/39069991) exposed nine image entries and one video in public slide metadata, with timestamps only. Image requests returned access refusals. Two attempts to load the public player through the available browser interface timed out. Slide contents remain unverified.

## Completed official seed-0 training

The official 200-epoch run completed with exit status 0. The terminal log contains every epoch from 0 through 199 exactly once, with all 267 optimizer steps completed per epoch and finite epoch-average losses. This totals 53,400 updates and 13,670,400 sampled windows after drop_last at effective batch 256. No checkpoint resume was used. Epoch-average training loss decreased from 1.2498 at epoch 0 to 0.2648 at epoch 199.

The trainer reported 9 h 48 min 38 s. The enclosing command took 9 h 48 min 43 s. Individual epochs took 106–212 s under the observed shared GPU workload. Peak logged PyTorch allocated memory was 13,477 MiB. The process-specific NVML monitor recorded a sampled peak of 15,250 MiB across 7,038 observations at five-second intervals. Peak host RSS was 1,483,868 KiB. Device-total memory includes unrelated work and is excluded from the process measurement.

The final checkpoint and retained epoch-199 snapshot each contain 2,597,632,689 bytes. The final training completion sentinel exists. Retained snapshots cover epochs 0, 25, 50, 100, 150, and 199. Checkpoint content and checksum adjudication are documented by the independent final-checkpoint audit.

The independent audit reloaded snapshots 150 and 199, checking 1,089 finite model, dual-EMA and optimizer tensors in each. The final live checkpoint, retained epoch-199 snapshot and wrapper manifest share SHA-256 `7047396663de22f02af0ae6676b579cb030be4c9effdd66729b093c22473f418`. The two final files are owner-read-only (mode 400). Evidence is `~/EEG-Loop/evidence/final-checkpoints.audit.json`. The tracked upstream server source remains clean at the pinned commit.

Machine-readable reconciliation is saved at `~/EEG-Loop/evidence/training-completion-r0.json`, including all 200 epoch-average losses. The training log SHA-256 is `0d8835fa63fb4e847abf9e7601ff19d4cfdefe5f1d5aa223401f9e737fa36a37`. Released-source train evaluation started automatically after the final checkpoint checks; its verified results follow below.

The complete raw training trace and observed per-epoch durations are archived with CSV, plotting source, PNG, editable SVG, embedded-font PDF and hashes in `figures/r0_training/`. All 200 loss values match the reconciliation record exactly. These plots describe the training objective and observed runtime on the shared GPU.

![Original JET seed-0 training objective and observed epoch durations](figures/r0_training/r0_training.png)

## Verified released-source train evaluation — A

The unchanged released wrapper evaluated the full processed train split with the frozen epoch-199 checkpoint, EMA1, evaluation seed 0, Gaussian noise scale 1, Heun 50 steps with an Euler final step, CFG 1 and batch 64. This is 99 conditional network evaluations per generated sample. Inference used float32 parameters without autocast. Both collections contain 68,445 event rows with the same class histogram.

| Collection/class | Rows in each collection | Distinct real waveform hashes | Released TS-FID |
| --- | ---: | ---: | ---: |
| Overall | 68,445 | 8,650 | **287.4627** |
| SPSW | 526 | 118 | 173.6802 |
| GPED | 10,963 | 2,397 | 265.8152 |
| PLED | 5,515 | 1,847 | 280.8499 |
| EYEM | 984 | 476 | 249.4804 |
| ARTF | 10,553 | 1,630 | 423.0609 |
| BCKG | 39,904 | 2,527 | 329.2736 |

Distinct waveform counts are within class and can overlap across labels. Unequal sample and waveform support affect class-score precision. The overall score is calculated from the combined feature distributions, rather than averaging the class scores.

The archive/reference verifier confirmed all 68,445 selected real rows, finite generated arrays, metric metadata and label counts. All seven independent numerical TS-FID comparisons passed the frozen tolerance. The verified claim is faithful execution of the released train-split command. Publication-result reproducibility remains undetermined because the exact Table 1 protocol lineage is unresolved. This result alone does not complete R0.

The original evaluation command took 2 h 57 min 12 s, including model loading, generation, real-reference collection, archive compression and metric computation. Sampled process NVML allocation peaked at 2,808 MiB across 2,119 five-second observations; peak host RSS was 30,779,780 KiB. Synchronized generation latency is reported separately below. The analytic leading-matrix estimate is 15,051,718,656 FLOPs per network evaluation and 1,490,120,146,944 per generated sample at 99 evaluations. Normalization, softmax, elementwise operations, solver arithmetic and data movement are excluded.

The 8,153,056,958-byte A EEG archive remains on the restricted server with SHA-256 `a1381b1557c4adb62b75efbafc67e874fb8e496a42dc7eace6bf08e851f96387`. Metrics SHA-256 is `3b138d0bc3b689cf900a43a027c98b9f9d0bba3e023a62147d2c1c8d53b3a015`. The completed archive, metrics and completion marker are owner-read-only. Aggregate metrics, numeric audit, integrity log, exact settings, resource evidence and hashes are copied to `results/r0-source-a-seed0/`; no EEG arrays were copied locally. `RESULTS.csv` and `RESULTS_README.md` record the verified track and cost definitions.

## Verified held-out evaluation — B

The frozen test command used the same epoch-199 checkpoint, EMA1, inference seed 0, Heun 50/99 NFE, CFG 1, batch 64, Gaussian scale 1 and float32 inference without autocast. Both collections contain 29,421 event rows. The full reference and generated-array checks passed, as did all seven independent TS-FID comparisons.

| Collection/class | Rows in each collection | Distinct real waveform hashes | Released TS-FID |
| --- | ---: | ---: | ---: |
| Overall | 29,421 | 3,243 | **365.1448** |
| SPSW | 567 | 46 | 329.9424 |
| GPED | 4,677 | 813 | 540.7131 |
| PLED | 1,998 | 704 | 240.8921 |
| EYEM | 329 | 120 | 277.0734 |
| ARTF | 2,204 | 528 | 835.5475 |
| BCKG | 19,646 | 1,176 | 365.8367 |

These are one-checkpoint, one-inference-seed measurements. The A/B populations differ in class composition, waveform support and recording distribution; their difference is not an isolated estimate of overfitting. Neither track has established the exact publication protocol. A is numerically 51.6027 (21.8785%) above 235.86, but that is not a controlled same-protocol performance comparison.

B took 1 h 47 min 21 s including reference collection, archive compression and metrics. Sampled process NVML peak was 2,808 MiB over 1,283 observations; peak host RSS was 15,481,248 KiB. The 3,504,274,444-byte archive SHA-256 is `9870912868b6fbcd812bcb43684a9fb4d6a6e0556cae2124a34fa48bc7bbf73c`; metrics SHA-256 is `25d401bb8b0d233a54b6f29f65d47663c2ff36c07e4872e8aad6aa0031d80745`. The archive, metrics and completion marker were frozen owner-read-only after successful evaluation and integrity verification. Aggregate evidence is under `results/r0-heldout-b-seed0/`; EEG remains remote.

## Synchronized generation cost

The registered profiler used two warmup and five measured batches of 64, synchronizing CUDA around each measurement on the shared A6000. Measured batch latencies were 12.5533, 13.1249, 13.1625, 13.0988 and 13.1021 seconds; median 13.1021 seconds. Median batch throughput is about 4.885 samples/s, or 0.20472 seconds/sample amortized across a batch. The latter is not single-request latency. Peak PyTorch allocated/reserved memory was 2,240.337/2,494 MiB, distinct from the 2,808 MiB sampled process NVML peak. Parameters were float32 and profile-process matmul TF32 was disabled. Earlier evaluation processes did not serialize their live backend flags; installed defaults and source inspection are documented in the configuration reconciliation. Shared-GPU measurements are not dedicated hardware benchmarks.

The leading-matrix estimate is 1,490,120,146,944 FLOPs per generated sample at 99 NFE. It counts two FLOPs per multiply-accumulate and excludes normalization, softmax, elementwise/solver operations and data movement. `generation-profile.json` records exact measurements and scope.

## Held-out signal diagnostics

The frozen Protocol B diagnostic procedure restores microvolts, uses all 16 named leads and preserves original event-row weighting. Full-set native diagnostics contain no all-zero windows, undefined primitive values or missing correlation edges in either collection. All 120 off-diagonal edges are defined in each class. Root independently checked class counts, every native finite/undefined count, frozen artifact hashes, 271,200 native/scale ratio scalars, all 200 primitive/family/class/macro summaries, residual-matrix MAEs and profile arithmetic. Evidence is `results/r0-root-review-v1/quality-review.json`, produced by `review_r0_quality.py` without accessing EEG arrays.

| Class | PSD | Band power | Morphology | Temporal | Spatial |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPSW | 3.3661 | 3.1300 | 2.3835 | 1.3090 | 3.6156 |
| GPED | 1.6893 | 2.4779 | 1.8449 | 2.5570 | 3.3820 |
| PLED | 1.3710 | 1.8986 | 1.5916 | 0.8372 | 2.6819 |
| EYEM | 2.5623 | 3.0825 | 3.0482 | 1.8499 | 5.1274 |
| ARTF | 5.9251 | 2.8892 | 1.6005 | 0.2026 | 3.8733 |
| BCKG | 4.9654 | 4.8702 | 3.0187 | 3.7831 | 11.5598 |
| Six-class macro | **3.3132** | **3.0581** | **2.2479** | **1.7565** | **5.0400** |

Values are mean dimensionless discrepancies normalized by per-primitive train/validation real-vs-real median scales, under the original equal-channel/primitive/class rule. They are neither significance levels nor clinical thresholds. The 200 fixed, overlapping within-class draws measure subset sensitivity, not independent training-seed replication or a population interval. Frozen calibration counts per class are 92/512/380/161/512/512. Limited distinct waveform support, particularly rare classes, remains relevant.

Native discrepancies also vary by lead: ARTF log10-PSD MAE ranges from 0.3593 to 1.2939, largest at F7–T3; its mean first-difference W1 is 26.9639 microvolts and centered-zero-crossing W1 is 0.17433. SPSW alpha log10-bandpower W1 averages 1.22923 across leads. Distances do not determine the sign of a power or amplitude change. These are statistical discrepancies, not evidence of clinical realism.

The separate, preregistered residual-channel diagnostic subtracts each domain's own full class/channel/time mean before within-window temporal centering. All covariance and Pearson summaries are defined, with zero near-constant/nonfinite counts. Six-class macro off-diagonal covariance MAE is **642.1078 microvolts squared**, and Pearson MAE is **0.27536**. Mean absolute class-mean off-diagonal residual correlations are:

| Class | Real | Generated |
| --- | ---: | ---: |
| SPSW | 0.33091 | 0.00728 |
| GPED | 0.32201 | 0.00201 |
| PLED | 0.26641 | 0.00364 |
| EYEM | 0.35464 | 0.02516 |
| ARTF | 0.16829 | 0.00427 |
| BCKG | 0.23467 | 0.00163 |

This aligns with the verified channel-separable architecture and independent channel noise. Finite-sample class-mean estimation shrinks residual variation; class counts match between domains. This remains a descriptive mechanism diagnostic, not a retroactive quality gate. Pure recurrence confined to each channel cannot introduce arbitrary class-conditional cross-channel dependence.

Aggregate quality evidence and exact hashes are in `results/r0-quality-evidence-v1/`. Class-mean EEG waveforms were removed from the mirrored residual summary and remain on the restricted server. The full native diagnostics, all calibrated repetitions, frozen scales and measured profile are preserved; no signal arrays or private reference indices were copied locally.

![Measured class-family discrepancies and residual channel association](figures/r0_quality/r0_quality.png)

The figure preserves exact source values in CSV with plotting code, PNG, editable SVG, embedded-font PDF and hashes under `figures/r0_quality/`. Its 200 overlapping subsets are descriptive repetitions; no confidence interval or independent-seed claim is shown.

## Four-part R0 conclusion and next action

1. Released-source model/data/training/evaluation consistency passed, with documented runtime and identity-evidence limitations.
2. The published numerical result was not independently reproduced; exact publication protocol lineage is unresolved. No claim that the publication is impossible to reproduce follows.
3. Measured A/B TS-FID, signal-quality discrepancies and resource costs are reported above, for one trained checkpoint and one generation seed per population.
4. Remaining discrepancies include publication settings, channel separability, repeated event windows, unverified patient identity, secondary-metric recipes and runtime differences. The investigated evidence identifies no configuration, integrity or arithmetic failure warranting a speculative rerun or score-driven evaluator change.

The historical research-lead decision permitted a bounded validation-only R1 after source-baseline freeze. The user subsequently required paper alignment first and prohibited author checkpoint use; these instructions supersede the earlier permission. All R1 implementation/probes/generation are stopped, with completed K=1 evidence and partial code preserved. The independent paper-aligned target prioritizes the main method/equations where the text conflicts internally, labels unspecified defaults, and freezes its implementation before results. Architecture/objective changes require new training from our own fresh initialization. No Loop or dataset expansion may proceed under the historical permission. The immutable `r0-baseline-final-v1` packet preserves the earlier point-in-time state; this live report records subsequent corrections.

