# R1 completed screen: research-lead decision

Date: 2026-11-12. Scope: our frozen released-source R0 checkpoint, five fixed validation arms. This decision does not concern the untrained joint-80-token/direct-vector paper-aligned model. No author checkpoint was used. This is a negative screen, not completion of the overall research mission.

## Observation

Independently inspected local aggregate copies under evidence/r1-screen-complete-v1: every arm configuration, metrics, numerical audit, native/weighted diagnostics and timing; continuation-filter; mechanism-probes/report; runtime. Checked probe/scorer implementation to establish estimands. All five numerical audits pass (35 overall/class checks). All six classes in every arm have zero all-zero reference/generated windows and 120 valid mean-correlation edges. Probe failed_k and failures are empty. All configurations refer to approved df72fb356fb197d4e59fd25a37bb8406e5b95a74a0eb575aa20b1ad216689bd3. The paired input-noise file is shared. This review uses aggregate evidence; it does not claim a new independent rehash of private EEG archives.

| Arm | TS-FID (lower better) | Matrix TFLOPs/sample | Median seconds/batch32 |
| --- | ---: | ---: | ---: |
| baseline | 365.6762855927624 | 1.490120 | 4.310376 |
| loop2 | 367.24184801640706 | 1.983907 | 6.504864 |
| loop3 | 386.4303986793036 | 2.477694 | 8.115149 |
| flow67 | 365.2480279759686 | 2.001879 | 6.756440 |
| flow83 | 365.0822349044729 | 2.483534 | 8.365210 |

Loop2 is worse by 1.56556 TS-FID versus baseline and 1.99382 versus its flow control. Loop3 is worse by 20.75411 and 21.34816 respectively. Extra-flow gains are small descriptive differences, not significant improvements. FLOPs exclude elementwise/norm/softmax/EMA-copy/solver work; equal block count is not equal total compute. Runtime reports no competing GPU process at capture; this is not proof of exclusive hardware throughout. Clean timing excludes probe forwards. Probe replay separately cost 0.8589 seconds and 30 extra suffix decodes.

| Family | baseline | loop2 | loop3 | flow67 | flow83 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSD | 3.50293 | 5.72310 | 10.41467 | 3.46679 | 3.45620 |
| Band power | 2.95657 | 5.81233 | 9.20822 | 2.93929 | 2.93417 |
| Morphology | 1.99699 | 9.52919 | 18.51045 | 1.99260 | 1.99140 |
| Temporal | 1.81254 | 6.74938 | 11.72523 | 1.81209 | 1.81199 |
| Spatial | 4.94082 | 4.84325 | 4.58431 | 4.94444 | 4.94546 |

Independently recomputed candidate/baseline ratios and class-family failure counts. Loop2 fails 20/30 class-family filters; loop3 fails 24/30. Both fail four macro families and the FID comparison. Their small spatial discrepancy reductions cannot offset the conjunctive failures and do not establish restored cross-channel dependence. These are full-screen exploratory weighted discrepancies, not count-matched Protocol B scores.

## Explanation

The tested intervention repeats blocks trained for a single pass, without recurrence training. The observed outcome is consistent with representation and endpoint drift under an out-of-training-distribution block composition. It is not an arithmetic failure or a nonfinite-state crash. K1 previously matched the original model exactly, including all 99 trajectory evaluations.

The twelve fixed real/noise-interpolation probes show final endpoint relative-L2 changes versus K1 averaging 0.525--0.979 for K2 and 1.115--1.929 for K3 across five times. Last interpass hidden relative changes remain 1.036--1.253 for K2 and 0.524--0.632 for K3, despite cosine similarities of about 0.975--0.998. Thus high directional similarity does not demonstrate a fixed point. Smaller relative increments at the third pass can partly reflect the enlarged denominator; they are not proof of convergence.

At t=.98 mean relative vector-field changes are 1.4614 (K2) and 2.8064 (K3), versus .0505 and .1072 at t=0. Endpoint-to-vector conversion and a changing reference-vector norm affect this relative measure: do not attribute the entire increase solely to the endpoint division. These are interpolation probes, not measurements along actual generated trajectories, and establish association rather than a causal decomposition of the final quality loss. A channel-separable wrapper cannot acquire conditional joint-channel dependence simply by repetition.

## Falsification

The preregistered experiment already tested the useful claim: repeating this fixed middle block group at inference improves generation at equal flow steps and competitive compute. Both paired extra-flow controls and signal-quality checks contradict that claim on this screen. More solver evaluations leave quality near baseline, whereas internal recurrence markedly degrades four families. This distinguishes the tested intervention from a generic claim that any extra computation is harmful.

No new generation is needed to establish this negative result. The current probe evidence does not identify a justified damping coefficient, alternate block boundary or K. Searching these now would be a new exploratory optimization, not verification of the failed hypothesis. A genuinely different, future falsifiable hypothesis would be that explicit recurrence training makes a shared core contractive or quality-improving under repeated application; it requires its own architecture, loss, compute-matched training controls and frozen validation plan. It is not authorized by this review.

## Result and limits

No selected arm. Both continuation filters are false. One own training seed and one paired noise realization cannot estimate training-seed uncertainty or establish a universal impossibility of recurrent EEG generation. The fixed 2,020-row validation screen contains only 23/18/141/50/69/361 distinct normalized waveforms and 7/2/3/5/11/38 linked recording components by class. Row counts overstate independent support; no p-value or generalization claim is warranted. Do not numerically equate this subset FID with full held-out R0 or paper235.86.

Separately inspected own-r0-rescore-v1/results.json: all 28 numerical checks pass. New-environment source1000 scores are A287.462731558105 and B365.1448073063774, reproducing frozen original scores within tolerance on identical saved arrays. Fixed paper-v1-1024 scores are A401.8720844316602 and B457.86343002666825. This rules out a material evaluator-runtime effect for these archives under the source operator; it does not test training-runtime effects. The feature-geometry change raises these values and does not reproduce235.86. Different feature spaces do not share a calibrated numerical scale.

## Decision

STOP this R1 screen with its negative result preserved. Do not run full validation, test evaluation, extra K, damping, alternative placement, R2 training or another dataset from this decision. Do not promote flow67/83 as a significant winning model.

Proceed with the separately preregistered own-fresh paper-aligned baseline, subject to actual materializer/storage/input-freeze and driver review. This remains unfinished baseline-alignment work, not a Loop rescue experiment. No baseline training result or 235.86 reproduction is promised. After its faithful execution and numerical/quality audit, reassess whether a mechanistically specified trained-recurrence experiment is warranted; freeze that proposal before results. Preserve all source-baseline evidence and keep the two architecture tracks separate. The overall mission remains active; expansion and statistical-improvement gates remain unmet.

## Additional provenance clarification

Root independently reopened the five private archives on the remote machine and verified their hashes, bit-identical frozen reference rows, exact labels, finite/nonzero 2,020 generated rows per arm, and unchanged source/operator/noise/checkpoint hashes. Its separate review is evidence/r1-v1/final-review-v1/review.json, SHA256104aad02b78eea9f9764c9f131238a4f98a58b3785760f6c36afe7cc302353c6. This archive-level verification is attributed to root; the research-lead inspection above used aggregate copies. Completed outputs are read-only.

The released train_tuev, train_tuab and train_tusz scripts train datasets independently; the paper does not report pooling the three datasets for Table1. There is no positive evidence that hidden three-dataset training explains235.86. Population differences between our A/B reports do not establish the cause of the publication gap. Its causal attribution remains unresolved; the separately aligned fresh training is still pending.
