# LATEST USER DIRECTIVE — align our baseline only

The user explicitly prohibits using the authors' checkpoint. Do not load, inspect, evaluate, initialize from or convert it. Preserve the unused download without using it. The active work is a separately named paper-aligned version of OUR reproduction, specified in PAPER_ALIGNED_V1_PROTOCOL.md. Original R0 and K1 evidence stay immutable. Loop remains held. This supersedes earlier checkpoint-inspection recommendations and the historical R1 permission below.

# USER OVERRIDE — publication alignment hold

The user's subsequent instruction “那你应该完全对齐啊” supersedes the earlier bounded-R1 authorization below. **R1 permission is withdrawn. No candidate generation, Loop implementation continuation, R2 or other-dataset work is authorized until the publication baseline is fully aligned and reviewed.** Completed R0 and K1 engineering evidence remain intact; the prior authorization is historical, not active. See ALIGNMENT_AUDIT.md for the paper/source/actual-run matrix, confirmed contradictions, unknown author settings and concrete evidence requests. Existing scores remain faithful released-source measurements, not a successful publication reproduction. Do not guess missing values or tune toward 235.86.

# Final R0 research-lead judgment

## Decision and authorization

**Released-source R0 is completed and audited. The published 235.86 result has NOT been independently reproduced. Bounded, validation-only R1 is authorized after this report and the final baseline evidence are frozen. No dataset expansion is authorized.**

This decision follows a fresh reading of the original user mission, rather than treating earlier audit shorthand as a new requirement. The mission requires that the baseline be implemented, trained, evaluated and audited; a report explicitly say whether the publication result is reproducible; and a reproduction failure be investigated before proceeding. R1 requires an audited baseline. It does not specify matching 235.86 as an R1 entry threshold. Demonstrable paper comparability and the 212.27 threshold are explicit expansion requirements. The prior provisional wording that unresolved publication lineage must necessarily prevent any R1 was too restrictive. This decision does not call publication reproduction successful or waive the expansion requirements.

The discrepancy has been investigated through unchanged source and data checks, checkpoint/actual-argument reconciliation, both predeclared evaluation populations, independent metric arithmetic, signal diagnostics, and public protocol searches. No implementation or arithmetic failure has been found that warrants another baseline run or a change of evaluator. Further seed repetitions would estimate variability but would not supply missing publication lineage. No routine user clarification or speculative rerun is required to start the limited R1 question already authorized by the mission.

## Four required judgments

1. **Consistency with the official model and data protocol: PASS for the released-source reproduction, with disclosed runtime differences.** The pinned 07f9e64 model, source preprocessing, complete data, 200 epochs/53,400 updates, loss, weighted sampler and primary EMA were retained. Both full archives pass reference/value/metadata integrity and all seven independent numerical checks. Actual arguments and final topology/dtypes show no detected mismatch. Python/PyTorch/CUDA/MNE versions differ from the README recommendation; official checkpoints omit args, so logged Namespace plus source supply those settings. Live training/inference TF32 flags were not all serialized; source/default/environment evidence is not equivalent to observing every live flag. The separate measured profile explicitly records matmul TF32 false.
2. **Independent reproducibility of the published result: NOT ESTABLISHED; this run did not reproduce 235.86.** Neither train-reference 287.4627 nor held-out 365.1448 can be asserted to use the exact publication protocol. Split/counts, 256-versus-250 frequency geometry, exact checkpoint and sampling lineage remain unresolved. Numeric differences are reported without pretending matched-protocol comparison. No claim that the publication is impossible to reproduce follows.
3. **Actual measurements: complete below.** These are one checkpoint and one generation seed per population. They establish a usable, reproducible comparison baseline, not robust superiority, clinical validity or statistical significance.
4. **Remaining discrepancies: documented, not silently corrected.** The source channel-separable architecture cannot model arbitrary class-conditional inter-channel dependence; residual diagnostics show a substantial corresponding deficit. Event annotation replication and copied recording content reduce distinct waveform support. Payload audit establishes no exact recording copies across splits, not verified patient separation. Secondary Silhouette/CBraMod settings are unavailable. Shared GPU affects measured runtime. Publication lineage remains unresolved and prevents publication-relative and expansion claims.

## Verified measurements

Astra directly read authoritative server A/B metrics, integrity logs, numerical checks, native lead diagnostics, calibrated summary and frozen hashes, residual scalar summaries, profile, and configuration reconciliation. EEG arrays and class-mean waveforms remained remote.

| Class | A train rows | A TS-FID | B held-out rows | B TS-FID |
| --- | ---: | ---: | ---: | ---: |
| Overall | 68,445 | 287.46273156 | 29,421 | 365.14480733 |
| SPSW | 526 | 173.68023189 | 567 | 329.94236186 |
| GPED | 10,963 | 265.81519977 | 4,677 | 540.71311299 |
| PLED | 5,515 | 280.84985235 | 1,998 | 240.89205777 |
| EYEM | 984 | 249.48040509 | 329 | 277.07344546 |
| ARTF | 10,553 | 423.06089525 | 2,204 | 835.54746930 |
| BCKG | 39,904 | 329.27355001 | 19,646 | 365.83667240 |

The larger B aggregate reflects a different real population/class mixture and separately generated archive; it is not an isolated estimate of overfitting. Pooled FID is not the mean of class FIDs. B symmetric overall is 365.1448073854919, difference 6.02248e-8; maximum class difference is 2.750317e-5, all below frozen tolerances. Requested bins remain 256 but actual feature dimension is 1000, as preregistered.

| Calibrated family | B mean normalized discrepancy |
| --- | ---: |
| PSD | 3.31321077 |
| Band power | 3.05806920 |
| Morphology | 2.24790449 |
| Temporal | 1.75645589 |
| Spatial | 5.04001414 |

These are ratios to primitive train/validation real-vs-real median scales, aggregated by the fixed procedure; they are not p-values or confidence bounds. BCKG spatial is 11.55984603. No generated/reference all-zero windows or undefined native primitives were found. Full-set native discrepancies vary substantially by lead: ARTF log-PSD MAE ranges 0.3593–1.2939 across leads, with F7–T3 largest; its per-lead first-difference discrepancy averages 26.9639 microvolts and zero-crossing-rate discrepancy averages 0.17433. SPSW alpha log-bandpower W1 averages 1.22923 across leads. These are discrepancies, not signed effects; they do not establish whether generated power is larger or smaller. Units were checked against the source: loader values times 100 recover microvolts, band-power distances operate on log10 integrated power, and covariance uses microvolts squared.

Residual-channel Pearson macro MAE is 0.27535677; covariance macro MAE is 642.10779714 microvolts squared. Both are defined for all six classes. The separately reported classwise absolute mean offdiagonal residual correlations are much smaller in generated signals than references. This is consistent with source channel separability and independent channel noise; repeating those same blocks cannot introduce cross-channel dependence. It is a limitation of the baseline and of pure channel-separable recurrence, not evidence that the implementation deviated from source. Later R1 may improve marginal spectral/morphological quality, but must not claim to repair this dependence mechanism.

Training took 9:48:38; source A command 2:57:12 and B command 1:47:21 (engineering completion reports). Separate synchronized profile: two warm-ups, five measured batches of 64; median 13.102068917 s/batch, allocated 2240.337 MiB, reserved 2494 MiB, float32, 99 NFE. Leading matrix estimate is 1.490120146944e12 FLOPs/sample; excludes elementwise, softmax, norms and solver work. Measurements share an A6000 with another job and are not dedicated-device benchmarks.

Final checkpoint SHA256: 7047396663de22f02af0ae6676b579cb030be4c9effdd66729b093c22473f418. B archive SHA256: 9870912868b6fbcd812bcb43684a9fb4d6a6e0556cae2124a34fa48bc7bbf73c. B metrics SHA256: 25d401bb8b0d233a54b6f29f65d47663c2ff36c07e4872e8aad6aa0031d80745. Final checkpoint/archives are preserved read-only by root. Calibration scale/position hashes match their frozen values.

## Bounded next action

After root freezes this completed report/evidence, Sol may implement only the initial R1 validation experiment: unchanged pretrained weights; blocks 1–4, repeated blocks 5–8, blocks 9–12; K=1/2/3 with no damping search initially. First prove K=1 output equivalence under identical inputs, conditioning, precision and random noise. Use validation only, fixed class/reference and paired noise seeds, no new test evaluation during design. Record hidden norms/drift, endpoint and vector-field changes across flow time. Compare the baseline 50-step sampler to equal-step loops and baseline extra-step controls at approximately matched block compute (K2 versus baseline67 steps; K3 versus baseline83 steps, with measured FLOPs/NFE differences disclosed). This initial authorization is one small falsification round, not a trainable-architecture search; review its evidence before damping changes or R2.

Freeze an explicit validation run specification before generating candidate outputs, including one shared initial-noise artifact/seed scheme and the same six-class histogram. Do not copy test-tuned settings or select classes from the B scores. Pure channel-separable loops cannot fix joint dependence; inspect whether extra hidden computation helps marginal generation at matched cost or instead destabilizes the pretrained network. Stop or revise on evidence, not on a desired conclusion. Additional data work, training, expanded method sweeps and confirmatory held-out comparisons remain outside this initial R1 authorization.

# Prior R0 audit record (historical)

The following sections preserve earlier findings and provisional OPEN decisions. Their pending-status statements describe when written and are superseded by the final four-part judgment above; source/data facts and frozen metric definitions remain unchanged.

# Research lead R0 audit

Status: complete training and final checkpoint audit verified by root/Sol; source-A evaluation and numerical audit passed, held-out B and diagnostics pending. Scientific exit judgment remains OPEN. This document does not authorize Loop implementation or experiments.

## Measured released-source A result

Astra directly read the authoritative server metrics, verification log and independent numeric audit through SSH after completion. Metrics SHA256 independently recomputed as 3b138d0bc3b689cf900a43a027c98b9f9d0bba3e023a62147d2c1c8d53b3a015, matching evidence/evaluation-integrity/train-seed0.metrics.sha256. Source archive: eval-released-train-seed0/; audit: evidence/evaluation-integrity/train-seed0.ts-fid-audit.json. The integrity log verifies generated values/metadata and all 68,445 reference rows.

| Population | Rows per collection | TS-FID |
| --- | ---: | ---: |
| Overall | 68,445 | 287.46273155668246 |
| SPSW | 526 | 173.68023188698768 |
| GPED | 10,963 | 265.81519976983776 |
| PLED | 5,515 | 280.8498523496012 |
| EYEM | 984 | 249.48040509432923 |
| ARTF | 10,553 | 423.0608952489155 |
| BCKG | 39,904 | 329.2735500113955 |

The independent symmetric covariance calculation passes all seven prespecified tolerances. Overall symmetric score is 287.46273162829345, absolute difference 7.1611e-8; maximum absolute class difference is 1.626735e-5 (SPSW). Effective feature dimension is 1000. This rules out the audited covariance-square-root arithmetic as a material explanation of the score, not every possible software or protocol discrepancy.

This is a faithful released-command measured result, not a demonstrated reproduction of paper Table 1. It is numerically 51.6027 above the paper's 235.86; that arithmetic comparison is not a matched-protocol performance claim. A uses training references, and paper lineage remains unresolved. Overall FID is a pooled distribution metric, not a weighted mean of class FIDs. ARTF has the largest observed class discrepancy, but class sample/support differences prevent treating class ordering as equally precise or sufficient to identify a failure mechanism.

Discrepancy investigation must first finish frozen B and diagnostics, then distinguish feature/split lineage uncertainty, signal-level mismatches, and one-seed training variability. Inspect already frozen source/config/checkpoint and calibration evidence for concrete errors; retain exact data multiplicities and sampler settings. Do not select a favorable split, checkpoint, feature dimension, guidance or solver setting to approach 235.86. No numerical-audit failure currently justifies changing the evaluator. Independent training repetitions, if subsequently warranted, must be labeled and authorized as reproduction diagnostics rather than a search for a publishable number. Held-out quality and scientific exit assessment remain pending; no Loop is authorized.

## Evidence reviewed

Reviewed pinned official source, PROTOCOL.md, BASELINE_REPORT.md, dataset and calibration reports supplied by Sol/root, and primary paper/poster materials. Root verified evidence/final-checkpoints.audit.json: epochs 150 and 199 each contain 1,089 finite model/EMA/optimizer tensors. Final live, retained and wrapper-manifest checkpoint SHA256 values agree: 7047396663de22f02af0ae6676b579cb030be4c9effdd66729b093c22473f418. Remote tracked source remains clean at 07f9e64. Sol reconciled epochs 0 through 199 exactly once, 53,400 updates, finite epoch means from 1.2498 to 0.2648, training time 9:48:38 and command time 9:48:43. Peak allocated memory was 13,477 MiB; sampled per-PID NVML peak was 15,250 MiB. These remote artifact checks were performed by root/Sol and reported to the research lead. Source-A TS-FID is now verified above; final B diagnostics remain pending.

The run preserves the released model, preprocessing, loss, sampler and complete 200-epoch schedule. Documented runtime differences remain Python/CUDA/package versions and GPU hardware. Dataset manifests support complete event extraction and exact-signal separation across released splits. Directory identifiers do not establish person identity: repeated EDF content spans distinct nominal subject directories, and anonymized headers lack a common cross-pool patient namespace. Report directory-level and exact-signal separation precisely.

## Evidence required before exit judgment

1. Final checkpoint identity, epoch 199, both EMA states, optimizer state, exact training arguments, completion record, 200 finite epoch records and immutable source/data/environment references. Distinguish model training time, total job time, shared-GPU throughput and peak allocator/device memory.
2. Separate final A/source and B/released-eval archives and metrics: 68,445 and 29,421 rows respectively, fixed class histograms, seeds, solver/EMA/CFG settings, source and checkpoint hashes, finite arrays, and successful archive/reference integrity verification. Pinned collect_ground_truth_batch uses torch.randperm and samples without replacement, failing if requested class counts are unavailable. With num_images=0 and match_gt, every processed file is selected exactly once. The frozen reference_index.py independently asserts full unique-index coverage and matching labels; this logic was re-read by the research lead. Unique files need not imply unique waveform content: report the separately audited within-split waveform multiplicities. Source-A archive/reference verification passed; corresponding B verification remains pending.
3. Overall and six-class released TS-FID, sample counts and symmetric independent numeric cross-check. Apply the frozen tolerance without tuning; flag any failure before interpreting magnitude. Rare-class finite-sample FID is not an equally precise estimate of class quality.
4. Native and calibrated signal diagnostics with frozen configurations, calibration scales, matched subset positions and undefined counts. The 200 overlapping subsets are sampling-sensitivity summaries, not 200 independent experimental replicates. Include separately registered residual-channel covariance/correlation diagnostics and named-lead results; these are descriptive mechanism checks, not retroactive gate additions.
5. Sampling latency, resource measurements and FLOP estimation scope. Separate shared-GPU measurements from dedicated benchmarking and theoretical FLOPs from wall-clock performance.
6. Explicit measured-result table and four-part judgment: released-source fidelity; publication-result reproducibility; actual metrics; unresolved discrepancies. Freeze all baseline artifacts before any later intervention.

## Unresolved publication comparability

- The public wrapper evaluates train, whereas Appendix D.1 calls its repeated 235.86 result held-out. Table 1's exact reference population and counts remain unconfirmed. Neither A/source nor B is automatically the publication protocol.
- Appendix A.2 describes 256 pooled frequency bins; released TUEV extraction retains 250 bins and four channel groups, dimension 1000. Exact publication feature geometry is unresolved.
- Table 1 checkpoint, training/evaluation seeds, generation count, sampler steps, guidance and checkpoint-selection policy have not been independently established. No usable official pretrained checkpoint was found.
- The architecture is channel-separable and exchangeable conditional on class; paper language describing inter-channel modeling does not match that mechanism. Preserve the source rather than silently adding cross-channel attention.
- Exact Silhouette clustering settings and downstream CBraMod augmentation protocol are unavailable from the released evaluator. Do not manufacture comparable secondary scores.
- The official poster repeats the score without resolving these details. Public slide contents remain unverified after access failures. No author contact or challenge bypass occurred.

Public metadata refresh reported by root at 2026-11-11T08:55:26Z: official GitHub issues collection (state=all, per_page=100) and releases collection both returned empty arrays; the default branch still resolves to pinned 07f9e6491796f4f2c717d6259b1a2e24afce6a77. Targeted primary-domain search recovered the previously reviewed paper, project page and README but no additional authoritative evaluation configuration. Root preserves the response record in public-protocol-check-v1.json. This records the checked public endpoints and search scope only; it does not establish that private, unindexed or otherwise unexamined materials are absent. Publication lineage therefore remains unresolved on current evidence.

## Repeated event windows and effective support

Root recomputed frozen reference-index coverage: all processed paths are unique and selected once, but normalized float32 waveform SHA256 support is much smaller:

| Split | Event rows / unique files | Unique waveform hashes | Unique hashes by SPSW / GPED / PLED / EYEM / ARTF / BCKG |
| --- | ---: | ---: | --- |
| Train | 68,445 | 8,650 | 118 / 2,397 / 1,847 / 476 / 1,630 / 2,527 |
| Validation | 15,487 | 1,028 | 23 / 18 / 151 / 50 / 69 / 749 |
| Test | 29,421 | 3,243 | 46 / 813 / 704 / 120 / 528 / 1,176 |

Per-class unique counts sum above collection-wide support because some identical waveforms occur with multiple class labels. Hashes describe exact equality after loader normalization and float32 conversion, not independent physiological observations. Neither event count nor unique-waveform count is an independent sample size: neighboring five-second windows can overlap, recordings share subjects, and nominal person identity is unresolved.

The research lead inspected the original duplicate audit: it tests train/validation overlap within repeated raw EDF groups, not within-split waveform multiplicity. Thus its zero-cross-split findings do not contradict these counts. The official preprocessor extracts every REC row as the full 16-channel window determined by start/end time; it saves the offending channel separately. The training dataset uses the waveform and class, ignoring the offending channel. Consequently, distinct channel annotations at the same recording/time produce identical model inputs, and different channel labels can assign multiple classes to the same full-window input.

Sol's completed provenance audit checked all 113,353 REC label joins and 1,553 actual pickle channel/label spot checks, with no mismatch. Redundant train rows decompose as 59,748 same-recording/same-coordinate annotations plus 47 copied-recording rows; validation as 14,449 plus 10; test as 26,136 plus 42. Multi-label waveform groups number 344/32/144 for train/validation/test. All 42 cross-path test waveform groups share full EDF sample payloads and extraction coordinates despite differing headers. The evidence attributes the observed multiplicity to released annotation-to-example semantics and copied recording content, not a newly introduced extraction bug. Official weights remain unchanged.

## Recording payload integrity and uncertainty clarification

Before formal metrics were available, Sol audited all 518 EDF sample payloads and their signal interpretation metadata, recorded under evidence/all-edf-payload-v1/. There are 481 unique payloads and 37 duplicate groups; all groups agree in signal order, channels, calibration and sampling metadata. No payload group crosses train/validation/test. This strengthens exact-recording separation beyond file-byte hashes, which can differ only because headers differ. It still does not establish independent patient identities or exclude nonidentical recordings from the same person.

| Split | EDF records | Unique compatible payloads | Directory IDs | Linked-directory components |
| --- | ---: | ---: | ---: | ---: |
| Train | 297 | 282 | 232 | 221 |
| Validation | 62 | 57 | 58 | 53 |
| Test | 159 | 142 | 80 | 63 |

Future uncertainty analyses shall treat connected components of nominal directory IDs linked by identical sample payloads with compatible signal interpretation metadata as indivisible clusters. Resample components with replacement and retain all their original event rows and multiplicities together; use the same component draw for paired model comparisons. Retain fixed class-macro estimands and report draws missing class support as undefined rather than silently dropping classes or inventing observations. The cluster is a conservative observable linkage unit, not a verified patient. Report class-supporting component counts and uncertainty limitations, particularly for rare classes. Inference repeats remain nested within checkpoints; independent paired training seeds are a separate variability level. None of the 200 overlapping frozen diagnostic subsets becomes an independent experimental replicate.

This is a pre-result clarification of grouping for uncertainty, not a change to TS-FID, reference counts, weights, primitives, margins or frozen subset selections. Train/validation linked components remain 221+53=274. Sol exported and root verified exact equality of all component membership sets with the frozen calibration, not merely equal counts; all eight exported checksum entries passed. Calibration configuration SHA256 is ade2110650e655ec1b4e6417a2eec0db35131bd9485c320bf97043436b1500da. No recalibration is required. Root also reconciled provenance CSV counts and coordinates against the aggregate in root-summary-check.json.

The held-out component map was frozen before any formal TS-FID result, SHA256 3c163e22183211050d271e1f09ba92e1040dfd9c4187011a9f29f133bbf7960e. Its 63 components contain 55 singleton directory IDs, four pairs, two triples, one group of five and one group of six, covering all 80 directory IDs. These are 63 observable linkage components, not 80 independent patients. Preserve calibration and subset artifacts. Held-out B archive integrity, metrics and diagnostic audits remain pending before the final exit judgment.

Preserve official row multiplicities. They weight reference moments and diagnostic empirical distributions; inverse-class sampling balances classes but retains within-class annotation multiplicity. Frozen calibration likewise samples original rows, and its requested counts are not counts of distinct waveform support. In particular, validation GPED has only 18 distinct inputs and validation SPSW 23; test SPSW has 46. Report this limited diversity alongside nominal sample counts. The previously passed component/count sparsity screen checks a different condition and does not establish rich morphological support or narrow uncertainty. No post-hoc threshold, deduplication, split change or test-driven selection is authorized. Future uncertainty must respect recording/linked-directory clusters and repeated observations rather than bootstrap event rows as independent samples.

## Decision rules

### Result-independent final adjudication framework

This framework is fixed while B remains in generation and before its score is visible. The final report must issue four separate judgments rather than a single ambiguous reproduction label:

1. **Released-source fidelity:** pass only if the pinned model/data/training semantics and final checkpoint are verified, documented runtime differences are retained, both final archives match frozen manifests/settings, and numeric audits pass. Source A currently satisfies its completed evaluation requirements; B remains pending. Runtime variation is a disclosed limitation, not proof of numerical equivalence to the authors' environment.
2. **Publication-result reproducibility:** established only with evidence linking the publication reference population, feature geometry, checkpoint/sampler/count settings and measured result. Currently indeterminate because that lineage is unavailable. A favorable B number alone cannot change this judgment; an unfavorable B number does not identify which unpublished setting differs. Distinguish inability to verify the publication result from evidence that no implementation could reproduce it.
3. **Measured behavior:** report both frozen A and B scores, all class/count/support information, calibrated and native signal diagnostics, residual association checks, and generation compute/profile evidence. One seed is a baseline measurement, not an estimate of robust method superiority. Describe physiologic limitations without using TS-FID as proof of clinical realism.
4. **Residual discrepancies and next action:** list unresolved lineage, runtime, patient-identity and finite-support limitations separately from detected implementation/numerical failures. Investigate concrete failure evidence before considering R1. Source fidelity does not silently substitute for demonstrated publication reproduction.

Mandatory remaining investigations are already queued: complete B reference/generated integrity and numerical cross-check; inspect all five frozen diagnostic families plus named leads and residual matrices for finite values, constants, amplitude/PSD/morphology or dependence failures; reconcile sample counts/support and diagnostic scales/subset hashes; inspect sampling latency/VRAM/FLOP evidence with shared-GPU interference disclosed. Check upstream final-checkpoint configuration against actual inference configuration and archived scale/label/montage settings, rather than tune them. No additional training run is mandated solely because a score differs from 235.86.

Additional same-baseline diagnostics are conditional on observed failure: trace a nonfinite/shape/count/hash inconsistency to its source, check unit transformations if amplitude or spectrum is implausible, and inspect representative class/lead waveform and PSD plots under a predetermined seed/selection rule if signal interpretation is unclear. Such diagnostics must preserve checkpoint, rows and evaluation settings; random sample pairing is not waveform reconstruction evaluation. Independent repeated baseline training can address seed variability if necessary, but cannot repair missing publication protocol evidence. Changing split, feature bins, EMA, solver, guidance, checkpoint epoch or data weighting to improve agreement is not an authorized investigation.

Completion of B does not automatically open R1. Astra must first inspect the full packet, issue the four-part report, and make a separate explicit decision against the mission's strict reproduction gate. If publication comparability remains unresolved, record that requirement as unresolved; do not call strict R0 successful merely because released-source execution passed. Until that decision, R0 remains OPEN and no Loop implementation or experiments are authorized.

Numerical proximity to 235.86 cannot resolve lineage; a lower value cannot establish publication reproduction. Distinguish faithful released-code execution from independently reproducing the paper's claimed result. If final metrics are poor or diagnostic failures appear, investigate against the already frozen protocol without selecting a more favorable split or feature extractor.

R0 is currently OPEN. Training completion is necessary but insufficient. Astra will issue a final evidence-based exit judgment after the actual evaluation and integrity/numerical/diagnostic reports arrive. No Loop implementation is approved by this audit. Any later expansion additionally requires the prespecified multi-seed, quality, compute and demonstrably paper-comparable thresholds; one baseline seed cannot establish them.

## Primary references

- Paper: https://proceedings.mlr.press/v306/wang26fp.html
- Source: https://github.com/Y-Research-SBU/JET/tree/07f9e6491796f4f2c717d6259b1a2e24afce6a77
- Official poster page: https://icml.cc/virtual/2026/poster/63823
