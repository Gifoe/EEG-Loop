# Environment alignment audit

R0 diverged from two explicit released installation choices. The pinned README prescribes Python 3.10 and the official PyTorch CUDA 12.8 wheel index. R0 used Python 3.11.5 and PyTorch 2.6.0+cu124. Its successful source execution establishes compatibility with that runtime, while exact README environment alignment remains incomplete.

| Component | Explicit released prescription | Observed R0 | Classification |
| --- | --- | --- | --- |
| Python | 3.10, patch unpinned | 3.11.5 | Different minor version |
| PyTorch distribution | Official cu128 wheel index | 2.6.0+cu124 | Different CUDA build |
| PyTorch version | requirements.txt says >=2.0 | 2.6.0+cu124 | Minimum satisfied, author version unknown |
| NumPy | Unpinned | 2.4.6 | Author version unknown |
| SciPy | Unpinned | 1.15.3 | Author version unknown; local pin preserves released sqrtm tuple API |
| MNE | Unpinned | 1.13.2 | Author version unknown |
| TensorBoard | Unpinned | 2.21.0 | Author version unknown |
| tqdm | Unpinned | 4.70.1 | Author version unknown |
| einops | Unpinned | 0.8.2 | Author version unknown |
| scikit-learn | Unpinned | 1.9.1 | Author version unknown |
| Matplotlib | Unpinned | 3.11.2 | Author version unknown |
| Efficiency measurement GPU | Paper Appendix B.3 specifies B200 | RTX A6000 | Different hardware |

The tracked release has requirements.txt and no environment YAML or dependency lock. The paper gives optimizer and model/data details but does not specify Python, PyTorch, CUDA or direct dependency versions. A missing version pin cannot support an assertion that R0 used the authors' package versions. In particular, the README supplies no MNE version recommendation. [Pinned installation source](https://github.com/Y-Research-SBU/JET/blob/07f9e6491796f4f2c717d6259b1a2e24afce6a77/README.md) [Pinned requirements](https://github.com/Y-Research-SBU/JET/blob/07f9e6491796f4f2c717d6259b1a2e24afce6a77/requirements.txt)

Installed NumPy, MNE, scikit-learn and Matplotlib declare Python >=3.11. Therefore changing only the interpreter cannot preserve this entire package set under Python 3.10. A separate prefix at ~/EEG-Loop/conda-readme-align-py310-cu128 can satisfy the explicit released constraints by resolving Python-3.10-compatible packages and PyTorch from the cu128 index. The resulting exact freeze must be recorded as a new resolver-selected environment. The release does not establish it as the authors' original environment.

The observed cu128 index offers PyTorch 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1, 2.10.0 and 2.11.0 builds. Selecting one requires an explicit alignment target and rationale. No version is inferred from the publication score. The candidate environment has not been installed.

The host driver is 560.35.03. NVIDIA documents CUDA 12.x minor compatibility for drivers >=525, with limitations for newer driver-dependent features and PTX. Therefore a cu128 build is not automatically excluded by this driver, and successful cu124 execution does not establish cu128 execution. An isolated CUDA smoke must verify the chosen build before a full run. Driver, system CUDA and global packages remain untouched. [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

The explicit training table agrees with R0 AdamW, 5e-5 learning rate, batch 256, 200 epochs, five warm-up epochs, betas (0.9,0.95), label drop 0.1, primary EMA 0.9999, logit-normal parameters (-0.8,0.8), and time epsilon 0.05. The source-normalized TUEV input is 16×1000 with amplitude scale 0.01. The checkpoint structure matches the released 12-block, hidden-768 configuration. Exact source/config evidence remains in evidence/config-reconciliation-r0-v1/. [Paper implementation details](https://arxiv.org/html/2605.21280v1#A2)

Publication alignment still requires resolving evaluation lineage. The paper describes 256 spectral bins and four regions, whereas the released cutoff/cap produces 250×4 features for TUEV. The released inference wrapper selects train references, while the paper describes held-out evaluation. Paper descriptions of inter-channel interactions also exceed the released channel-folded implementation. Original preprocessing stores float64 signal arrays, while the paper describes float32 pickles; the loader subsequently casts model inputs to float32. These distinctions must be adjudicated against the paper and our own source implementation, with each conclusion traced to explicit evidence. Environment alignment alone cannot resolve them.

The user has directed all alignment work to our own reproduced checkpoint. The author checkpoint transfer was stopped by terminating the exact owned SCP process. Its CPU inspector never ran, and no author tensors were loaded, evaluated or used for initialization. The unused public download and remote partial file remain preserved.

R1 has no active remote preparation, gate or screen jobs. Completed K1 evidence and fixtures remain intact. Two partial screen source files are preserved locally. No candidate arms were generated. New full training, evaluation and sensitivity runs await Astra's review of one concrete paper-aligned target. Unspecified author details may retain documented released defaults. The target must separately adjudicate token layout and positional embedding trainability, full-time versus patchwise moment/TV losses, and spectral feature geometry before implementation. Existing source, weights, preprocessing and measured R0 results remain preserved under their original labels.

Initial machine evidence is at evidence/environment-alignment-v1/environment-alignment.json; the user-override transport and no-author-weight-use status is recorded in evidence/environment-alignment-v2/, with source hashes, installed Requires-Python metadata, prescribed-versus-unknown classifications, and the observed job/driver state.


