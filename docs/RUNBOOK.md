# Executed protocols and reproduction layout

This repository is an archive of the code and completed aggregate evidence. It does not include EEG payloads, private row indices or checkpoint payloads. No experiment is automatically resumed by cloning or validating it. The research run remains paused.

## Preserve the upstream identity checks

The frozen research drivers deliberately verify the original upstream Git HEAD and a clean tracked tree. The new archive repository has its own commit history. To execute these drivers, use a separate pinned upstream checkout and overlay the experiment directory rather than removing those checks:

```bash
export EEG_LOOP_ROOT=/absolute/path/to/run-root
export EEG_LOOP_CODE=/absolute/path/to/EEG-Loop
mkdir -p "$EEG_LOOP_ROOT"
git clone https://github.com/Y-Research-SBU/JET.git "$EEG_LOOP_ROOT/JET"
git -C "$EEG_LOOP_ROOT/JET" checkout --detach 07f9e6491796f4f2c717d6259b1a2e24afce6a77
cp -a "$EEG_LOOP_CODE/experiments" "$EEG_LOOP_ROOT/JET/"
cd "$EEG_LOOP_ROOT/JET"
```

The root upstream source files archived here are byte-identical to that pinned commit. Our frozen experiment source bytes are retained as recorded; `.gitattributes` disables line-ending conversion so their hashes survive Git transfer.

## Environments

Executed on Linux/NVIDIA RTX A6000 with isolated Miniconda prefixes:

| Track | Python | Torch | NumPy | Precision |
| --- | --- | --- | --- | --- |
| Released-source R0 and fixed R1 | 3.11.5 | 2.6.0+cu124 | 2.4.6 | BF16 training / FP32 inference |
| Paper-main-method interpretation | 3.10.19 | 2.7.1+cu128 | 2.2.6 | BF16 training / FP32 inference |

Exact captured packages are in `results/environment/source-r0/` and `results/environment/paper-aligned/`. The latter also has the conda explicit lock, wheel provenance and actual import/version check. These are the runtimes we used, not a recovered author environment.

Example environment restoration, from the archive checkout:

```bash
conda create -p "$EEG_LOOP_ROOT/conda-r0" python=3.11.5 -y
conda run -p "$EEG_LOOP_ROOT/conda-r0" python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r "$EEG_LOOP_CODE/results/environment/source-r0/pip-freeze.txt"

conda create -p "$EEG_LOOP_ROOT/conda-paper-aligned-v1" python=3.10.19 -y
conda run -p "$EEG_LOOP_ROOT/conda-paper-aligned-v1" python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$EEG_LOOP_CODE/results/environment/paper-aligned/pip-freeze.txt"
```

## Released-source baseline

Use the licensed TUEV v2.0.1 EDF root containing `train/` and `eval/`. The released preprocessor produces `train/val/test` with 68,445/15,487/29,421 event rows for the audited corpus. Counts include repeated waveform annotations and do not represent independent patients.

```bash
export TUEV_EDF=/absolute/path/to/tuh_eeg_events/v2.0.1/edf
conda activate "$EEG_LOOP_ROOT/conda-r0"
cd "$EEG_LOOP_ROOT/JET"
python data/preprocess_tuev.py --input-dir "$TUEV_EDF" \
  --output-dir "$EEG_LOOP_ROOT/tuev-official"
bash scripts/train_tuev.sh "$EEG_LOOP_ROOT/tuev-official" "$EEG_LOOP_ROOT/baseline-seed0"
```

Actual training used the unmodified 12-block/768-width/12-head configuration, effective batch256, AdamW5e-5,200epochs,267steps/epoch,53,400updates, seed0 and dual EMA. The original full trainer log is `results/source-r0/training.log`. Monitoring/checkpoint wrappers are in `experiments/looped_jet_tuev_v1/run_r0_*.sh`.

Explicit held-out evaluation with an **own-trained** output directory:

```bash
python inference.py --dataset tuev --datasets_dir "$EEG_LOOP_ROOT/tuev-official" \
  --resume "$EEG_LOOP_ROOT/baseline-seed0" --output_dir "$EEG_LOOP_ROOT/own-eval-test" \
  --eval_split test --eval_label_mode match_gt --num_images 0 --gen_bsz 64
```

Changing `--eval_split` to `train` records the separately labeled source-script population; do not combine its score with Test or validation-subset scores. Exact captured configurations, integrity checks and original metrics are in the result subdirectories. Do not download or initialize from author weights for these protocols.

## Frozen paper-aligned and Loop drivers

The complete code is under `paper_aligned_v1/` and `r1/` in the experiment module. Frozen protocol documents are beside them. The data/reference/calibration artifacts used as private inputs are not distributed here. Regeneration requires the licensed corpus and the audit/index/calibration tooling in the experiment module. Captured manifests describe the original run and original paths; they are evidence, not portable substitutes for those files.

The executed aligned sequence was environment/synthetic/CUDA smoke checks → float32 storage view/readback/full mapping audit → input freeze → ownfresh seed0 full200epoch training → fixed full Test evaluation. Its Train-reference comparison was then stopped at user request. `paper_aligned_coordinator.py` is the archived execution sequence, not an instruction to resume a terminated process. Source/input/storage guards must be satisfied before any newly authorized run.

The joint model implements the explicitly frozen main-method interpretation in `PAPER_ALIGNED_V1_PROTOCOL.md`; unspecified and internally conflicting author settings remain declared interpretations. `results/paper-aligned/seed0/` contains all200epoch records and completion/checkpoint metadata. `results/paper-aligned/test/` contains completed full-test metrics, quality, profile and both numerical audits.

The fixed R1 sequence was audited source baseline → exact K1 equivalence → frozen2020validation rows/noise → baseline/Loop2/Loop3/67step/83step arms → native/calibrated diagnostics, numeric and archive integrity checks → STOP decision. All five arms completed. The sourceR0 checkpoint was unchanged. No trainable recurrence or subsequent variant was run.

For portability, regenerate evidence under a new explicitly reviewed configuration rather than editing hardcoded historical hashes merely to pass the guards. The figure scripts and historical reports likewise record original paths; their aggregate inputs are archived for inspection.
