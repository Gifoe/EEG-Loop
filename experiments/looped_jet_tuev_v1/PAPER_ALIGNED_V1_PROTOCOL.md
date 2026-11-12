# Paper-aligned JET v1: frozen implementation target

Status: implementation specification under research-lead review, before any new score. The user's instruction authorizes aligning our reproduction and prohibits using the authors' checkpoint. **No official checkpoint may be loaded, inspected, evaluated, converted, or used for initialization.** Original source-R0 artifacts stay immutable. Loop remains held.

This is a separately named paper-aligned implementation derived from our source reproduction, not a claim to recover undisclosed author settings. Where the paper conflicts internally, the primary method and equations take precedence over efficiency prose; where it is silent, retain compatible released defaults and label them. No search over seeds, cutoffs, checkpoints or loss weights is permitted.

## Exact model target

Keep 16 bipolar channels, 1000 samples, patch200, six classes, width768, depth12, heads12, existing block/adaLN/time/class embedding and final-projection implementations. Patchify to [B,16,5,200], project200→768, then flatten in channel-major/time-minor order to [B,80,768]. The same attention sequence spans all 80 tokens; do not fold channels into batch. Compute one condition vector per example and supply it to all 80 tokens. Invert the exact flattening on output.

Use a learnable positional Parameter of shape [1,80,768], initialized with the existing 1D sine/cosine function over token indices0…79; requires_grad=True. No new channel embeddings or architectural additions. This initialization is an explicit deterministic implementation choice, not a reported author value. Main §3.2 specifies C*N tokens and learnable positions; AppendixB.3's five-token efficiency description conflicts and is not used to override that method. Measure actual parameters/FLOPs rather than asserting the unchanged rounded129.9M count.

## Flow and losses: main equations take precedence

The network directly predicts v_theta(z,t,c), as §3.1 describes. Sample independent unit Gaussian e; training t=sigmoid(N(−0.8,0.8)), then clamp only its lower bound to0.05. Set z=t*x+(1−t)*e and x_hat=z+(1−t)*v_theta. Use the following single objective, with no additional velocity MSE, STFT or spectral term:

- Reconstruction: mean absolute x−x_hat across batch,channels,time, coefficient1.
- Statistics: compute each channel's mean and sample standard deviation over all1000points, then mean absolute real/predicted mean differences plus mean absolute std differences over batch/channels, coefficient1. Retain PyTorch correction=1 explicitly because the paper does not specify a different convention.
- TV: sum abs(x_hat[...,1:]−x_hat[...,:-1]) over all999consecutive differences, divide by T=1000, then average batch/channels, coefficient0.1. Include all four former patch boundaries.
- Pearson: retain source's per-example flattened16×1000 centering/correlation, denominatorepsilon1e−8 and clamp[−1,1], averaged across batch, coefficient0.1. Paper does not specify its correlation axes; this is a retained default, not an asserted author fact.

Inference integrates v_theta directly with the unchanged uniform Heun50 grid and Euler final step (99NFE), CFG1, unit Gaussian initial noise and EMA1. There is no endpoint-to-vector division and no denominator clamp. The Table9 label “Min Time” is interpreted as the lower training-time bound above; the paper does not explicitly define that operational detail. This declared choice makes the main vector/endpoint equations internally consistent and differs from released endpoint prediction with denominator clamping. Do not represent it as independently verified author code. Inference includes t=0 as required by the noise-to-data ODE, despite the minimum training-time value.

These model/loss changes require **new training from scratch with our own seed0 initialization**. Reusing our endpoint-model checkpoint as if it had been trained under this objective would be invalid. Neither official weights nor warm-started own weights are part of v1. Our existing checkpoint remains the untouched released-source comparison.

## Data and training settings

Use the existing audited TUEVv2.0.1 raw recordings, exact source montage/filtering/window extraction and official train/validation/test memberships. Preserve row multiplicities and metadata. PaperA.1 states preprocessed float32 arrays: create a separately named view from current audited preprocessed signals by rounding microvolt arrays tofloat32 before loader normalization. Retain the released loader operation unchanged: data / 100.0, then conversion to torch.float32. With the float32 stored array this division remains float32 in the resolved NumPy runtime; assert and record that actual dtype. Do not insert a float64 upcast to emulate the old float64-storage arithmetic. The intended change is storage precision with the original loader operation retained. Preserve original float64 source pickles; record both raw and loader-space hashes, counts and before/after maximum/RMS differences. Do not re-filter/resample with a different dependency version without a separately declared change. Store the new view once; do not modify original data.

Paper does not fully specify filtering, split memberships, duplicate-event policy or library versions. Retaining their audited source definitions is a declared choice, not proof of author parity. The new view's mapping must be one-to-one with all113353rows and preserve labels/montage/order. Use validation for diagnostics and eventual selection; fixed200epoch finalEMA1 is the only scored checkpoint, so no validation/test checkpoint search.

Train200epochs,batch256,effectiveLR5e−5,warmup5epochs,constantafterwarmup,AdamWbetas(.9,.95),weightdecay0,labeldrop.1,EMA.9999/.9996,weightedreplacement sampler inverse class count(alpha1),Ndraws/epoch,drop_last,seed0. Retain BF16 training/float32 inference. Fix actual backend flags and package versions in the run manifest before training. These unspecified defaults remain disclosed.

Environment: isolate Python3.10 and the README's official PyTorch cu128 distribution, with compatible dependency builds resolved and locked by Sol before any new score/training. The README supplies no exact Torch/MNE/etc version pins, so the resolved lock is our reproducible realization of its instructions, not an invented historical author lock. Run import/CUDA/backward/sampler smoke on the available A6000 before launch. Do not change system drivers or other jobs. If batch256 fails, report measured failure and get a research-lead implementation decision; do not silently substitute a smaller effective batch or model.

## Fixed aligned metric v1

Define a new metric identifier `tsfid-paper-v1-1024`, preserving original `tsfid-source-1000` results unchanged.

Input is loader-normalized [N,16,1000]. Castfloat64; calculate rFFT magnitude retaining all501coefficients (0–100Hz inclusive). **Disable optional high-frequency truncation**, rather than guessing a missing cutoff. Adaptive-average pool the spectral axis to exactly256bins; pool channels into the same four contiguous source-montage groups; flatten to1024features; log1p; per-example mean subtraction and sample-std normalization, floor1e−6. Use source sample-covariance convention and independent symmetric numeric audit at tolerance0.001+0.00001*abs(score).

Full-spectrum pooling is chosen before scores because the paper makes truncation optional and specifies256bins. The preprocessor's75Hz low-pass remains unchanged. No post-score choice among50/75/100Hz cutoffs or bin counts is permitted. Spatial-group layout and normalization/covariance details retain source defaults where unspecified. A 1024-dimensional score is **not directly interchangeable with** the frozen1000-dimensional score or established as the author's exact Table1 metric.

Evaluation-only diagnostic: after source/config freeze and synthetic feature tests, recompute this new metric on our existing A/B archives, with no waveform regeneration or author weights. Label the output as “our source-trained R0 archive, re-scored under aligned-metric-v1,” retaining the source metric beside it. This isolates evaluator geometry without claiming to repair training. It cannot select metric settings or determine whether to run the already fixed target.

For the new trained model, primary population is full official held-out test29421rows with match_gt and no replacement, seed0,batch64,EMA1,Heun50/99NFE/CFG1. Secondary training-reference evaluation68445rows uses the same fixed settings, solely as labeled source-protocol comparison. Report both metric identifiers on each archive. Publish exact counts/histograms, source/config/checkpoint/feature hashes and independently audited scores. Freeze metric/evaluation code before either old-archive rescoring or new-model results.

## Engineering checks before execution

1. Verify input/output reshape identity and token order; assert attention sequence80 and batchB, learnable positional shape/gradient, block count/head count and actual parameter total.
2. Test full-time statistics on a constructed signal whose patch means differ; require the implementation to equal direct1000point reference calculations, not patchwise results. Test TV with jumps only at patch boundaries and require all four to contribute. Check explicit1/T normalization.
3. Test x_hat=z+(1−t)*v algebra at t=0,.05,.5,.98,1; direct vector inference remains finite with no1/(1−t) denominator. Check loss coefficients and absence of extra terms, conditioning/class-drop and EMA update semantics.
4. Test metric shape1024, full501bin support and four-group mapping using synthetic frequency/channel fixtures; independent numeric equality on identical and perturbed collections. Preserve original1000feature extractor unchanged.
5. Freeze one-to-one data-view mapping and its changed hashes; labels/counts/components unchanged. Reconcile code, isolated environment lock and backend flags before a fresh full200epoch run.
6. Smoke model memory/speed/finite backward at requested batch256 before launch. Preserve smoke artifacts separately. Full baseline training and old-archive metric-only work may proceed only after research-lead review of these concrete tests and frozen target artifacts.

## Reporting and limits

Variant name: `jet-paper-aligned-v1`. Enumerate its deltas from `jet-source-r0`: jointtokens,learnableposition,directvectorobjective/minimumtrainingtime,full-time moment/TV reductions,float32 storageview,README-compatible environment,and1024feature metric. Report fixes individually; the experiment tests their coherent combined specification, not causal attribution of each change. No ablation or tuning is authorized here.

The target prioritizes explicit method/equations and documents interpretations for contradictions/omissions. It can establish reproducibility of this frozen paper-aligned interpretation; it cannot retrospectively establish the undisclosed author's exact configuration merely by approaching235.86. User-required baseline alignment and all Loop holds remain subject to lead review of the resulting evidence. Inquiry draft stays unsent; no official checkpoint use or other-dataset work.
