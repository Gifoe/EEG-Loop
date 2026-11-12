# R1: fixed-weight recurrent computation on the audited source baseline

Status: specification prepared; implementation and execution wait for root's confirmation that the completed R0 evidence/report packet is frozen. This file does not modify PROTOCOL.md or the published-result/expansion gates.

## Question and interpretation

Test whether repeating a pretrained internal Transformer segment improves marginal EEG generation relative to unchanged JET and extra flow integration at approximately matched compute. The baseline is the audited released-source reproduction, not a claimed reproduction of the publication's 235.86. Original test results are descriptive R0 evidence and must not select R1 settings, samples, classes or outputs.

This is a single-checkpoint, single-noise-seed exploratory validation screen. No inferential significance, paper-relative superiority, clinical validity or dataset-expansion claim is permitted. Source channel separability remains unchanged; recurrence cannot introduce class-conditional cross-channel dependence. Improvement in spectral distributions alone does not establish realistic multichannel physiology.

## Immutable inputs and architecture

- Official source: 07f9e6491796f4f2c717d6259b1a2e24afce6a77.
- R0 epoch-199 checkpoint SHA256: 7047396663de22f02af0ae6676b579cb030be4c9effdd66729b093c22473f418; load EMA1 without modifying the checkpoint.
- Preserve preprocessing, montage, label mapping, five temporal patches per channel, time/label conditioning, and channel-folded forward semantics.
- Prefix: zero-based blocks 0:4. Core: blocks 4:8. Suffix: blocks 8:12. Apply the complete four-block core K times, reusing the same trained weights and conditioning each time, then apply suffix and original final normalization/projection once.
- No additional learned parameters, optimizer, retraining, cross-channel operations, damping, anchors, hidden normalization, dropout changes or adaptive routing. Patch/time/label embeddings are computed once per network evaluation, not reinjected between core passes.
- Float32 inference, no autocast, eval mode, CFG 1, Gaussian noise scale 1, t epsilon 0.05. Record actual backend flags, disable matmul TF32 to match measured R0 profile, and retain other source backend defaults. Batch size 32 for every arm including final partial batch.
- Retain official Heun integrator with its Euler final step and uniform time grid. Additional flow-step controls change only the number of steps.

## Fixed validation population and noise

Use official val only. Freeze a stable sorted processed-file index and verify labels/hashes against existing validation metadata. For classes 0 through 5, select min(512, available rows) without replacement using one NumPy default_rng(20261112), consuming draws in ascending class order. Sort selected dataset indices within each class, then concatenate classes in ascending order. Expected counts are 119/291/512/86/500/512, total 2,020. If observed counts differ, stop for data/config reconciliation rather than substitute another population.

Freeze selection code, NumPy version, ordered selected-file hashes, class schedule and index hash before candidate generation. Use all rows with their original multiplicities; do not deduplicate or pick visually representative samples. Validation has only 23 unique SPSW and 18 unique GPED waveform inputs in its entire class pools, so even these nominal counts overstate independent support. Report selected unique waveform counts and contributing payload-linked directory components. Keep private identifiers/signals on the server.

Generate one initial Gaussian noise tensor on CPU with torch.Generator(device='cpu').manual_seed(20261112), torch.randn((2020,16,5,200), dtype=float32), using the frozen label order. Save it once with a hash and reuse the same tensor rows for all five arms, transferred to the GPU in identical batches. Do not regenerate per-arm noise or rely on differing RNG consumption. Record PyTorch version and the actual tensor hash. No conditioning labels may be inferred from generated outputs.

The prefix/core instrumentation probe uses the first two selected rows of each class and a separate CPU Gaussian tensor from seed 20261113. Freeze these twelve indices and noise before running probes.

## K=1 equivalence gate, before candidate screening

Compare the unmodified official model forward and the recurrent wrapper at K=1 using identical EMA1 parameters, inputs, labels and precision. For each of twelve fixed probe examples use t in {0,0.1,0.5,0.9,0.98} with z=t*x+(1-t)*epsilon. Test both normally conditioned labels and the official null-conditioning label, although screen CFG remains 1.

Require torch.allclose(output_original, output_K1, atol=1e-6, rtol=1e-5) for every probe, all outputs finite, and identical shapes. Report maximum absolute error and relative L2 error (denominator max(norm(original),1e-12)), not merely a boolean. Also check state-dict keys and hashes before/after inference to rule out mutation.

Run an unmodified and K=1 full 50-step trajectory on the first six fixed probe rows using identical saved initial noise and label order. At every actual solver network evaluation and final waveform require the same atol=1e-6/rtol=1e-5 comparison; report per-time and final maximum errors. This is a short batch, not a shorter solver. All EMA handling and network wrappers must preserve official results. Any failure blocks K=2/3 generation until the implementation error is explained and fixed; do not loosen tolerance or proceed with a merely similar K=1.

## Five predeclared arms

Execute all arms below after equivalence passes, unless a hard numerical failure prevents completion. Run sequentially in listed order with no other R1 model jobs. Keep other-user GPU jobs untouched and log shared-device contention.

| Arm | K | Flow steps | NFE | Transformer block evaluations/sample |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1 | 50 | 99 | 1188 |
| loop2 | 2 | 50 | 99 | 1584 |
| loop3 | 3 | 50 | 99 | 1980 |
| flow67 | 1 | 67 | 133 | 1596 |
| flow83 | 1 | 83 | 165 | 1980 |

loop2 versus flow67 differs by 12 block evaluations (about 0.76% relative to loop2); loop3 versus flow83 matches block count exactly. These are approximate compute matches because embeddings, conditioning, final projections, solver work and EMA copies occur per network evaluation. Report leading-matrix FLOPs from actual arm shapes/operators plus measured latency and memory; do not equate block count with total FLOPs. Equal-step comparisons are loop2/loop3 versus baseline.

## Instrumentation and mechanism probes

Use the fixed twelve-example t-grid probes for K=1/2/3. At prefix exit, each core-pass exit and suffix exit record per-example hidden RMS, maximum absolute value, and finite status. Between core passes record relative L2 change with denominator floor 1e-12 and cosine similarity; define zero-norm cases explicitly rather than imputing similarity. Preserve class and time breakdowns.

For each core pass, send that pass's hidden state through a diagnostic copy/use of the same suffix and final projection to obtain an intermediate endpoint prediction. This extra decode is for probes only, must not alter hidden states or candidate generation, and its cost must be reported separately. Record endpoint differences versus K=1 and resulting vector-field differences v=(x_pred-z)/max(1-t,0.05), both absolute RMS and relative L2 with denominator floor 1e-12. Retain a small private set of intermediate waveform predictions for fixed examples, with summaries/plots generated only from those preselected rows. No sample hunting from test or best-looking outputs.

For actual generation, record hidden/flow diagnostics for the fixed probe rows or an equivalent separately timed probe replay; exclude instrumentation overhead from clean generation timing and report it separately. Avoid recording all hidden tensors for all samples. Report dependence on t and whether repeated computation amplifies, attenuates or destabilizes the vector field. Norm growth alone is not a stop criterion or proof of pathology.

## Metrics, timing and pilot filters

For every completed arm compute released-source TS-FID overall and per class on the exact 2,020 real/generated rows, with effective 1000-dimensional features and independent symmetric numeric cross-check at the frozen R0 tolerance. Use the same fixed real references across all arms. Preserve native metrics/archives and hashes. Do not compare this validation-subset score numerically to paper235.86 or R0 full test365.1448 as if populations matched.

Compute the same five native signal-diagnostic families used in R0, with identical units and primitive definitions. For dimensionless weighting, divide each primitive/class/channel by its existing frozen train/validation calibration median, then average channels, primitives per family and six classes equally. Use the entire fixed screen once per arm; do not reuse the 200 B subsets. This weighting does not make the screen count-matched to B calibration: counts differ, so label these as exploratory validation-weighted scores, not the Protocol B matched calibration or final quality gate. Native per-class/per-lead results remain visible.

The deterministic continuation filter for loop K is conjunctive:

1. Its overall TS-FID is strictly below baseline AND its corresponding extra-flow control (loop2 versus flow67, loop3 versus flow83).
2. Every one of its five macro family scores is at most 1.10 times baseline's same-family score.
3. Every class/family score is at most 1.25 times baseline's corresponding score.
4. All outputs and required metrics are finite, no generated all-zero window, and no required class/primitive is undefined.

Quality-filter comparator is baseline for both K values; extra-flow controls are additional TS-FID/compute comparators and must also have all quality scores reported. Do not introduce the real-vs-real additive margins from the count-matched B gate into this unmatched-count pilot. These pilot filters do not replace final significance, nondegradation or expansion criteria. If baseline has a zero family score, only zero candidate discrepancy meets that relative filter; do not add an unregistered epsilon to pass it.

Record total generation time, synchronized median batch latency after two warm-ups and five measured batches on fixed saved-noise rows, peak allocated/reserved VRAM, backend flags, device sharing, NFE, block count and leading-matrix FLOPs. All arms use batch32 and the same precision. If a model exhausts memory, stop and report; changing batch size for all arms requires a documented implementation amendment before resumed comparisons, not silently different per-arm sizes.

## Failure handling and next decision

Nonfinite generated/model states, hash/shape/count mismatch, all-zero generation or failed numerical metric checks are hard failures for that arm; retain evidence and do not compute a favorable score from remaining rows. Complete safe controls when useful, but never patch out failed windows. A numerical failure in one loop is evidence about that intervention; failure of K=1 equivalence is an implementation problem and blocks both loops.

After all valid arms, Astra reviews Observation → Explanation → Falsification experiment → Result → Next decision. If neither loop passes all continuation filters, stop this screen and analyze the negative result; no automatic damping, alternate block placement, extra K sweep or R2 training is authorized. If both pass, select the lower overall TS-FID, breaking an exact tie by smaller K. This selection is provisional, with no significance claim.

Any full-validation follow-up requires Astra's review of the complete five-arm packet and measured cost first. No automatic full-validation or new test run is dispatched by this specification. Further recurrence variants or training need a new falsifiable mechanism and explicit lead authorization. Final held-out evaluation is reserved for frozen selected models; neither classwise B weaknesses nor B scores may tune the intervention. TUSZ/TUAB work remains prohibited under the unmet expansion gate.

## Required artifacts

Store under a new R1 directory: protocol/source hashes; immutable R0 checkpoint reference; selection and noise manifests; K1 equivalence report including trajectory errors; per-arm configuration/log/completion status; archives and native/calibrated-weighting metrics; numeric audits; class/time hidden and vector diagnostics; timing/FLOPs/memory; and a five-arm table distinguishing successful, failed and unrun arms. Keep the R0 checkpoint/source/evaluation evidence unchanged. Upload no private EEG or identifiers to public repositories.
