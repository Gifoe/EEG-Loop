# R0 configuration reconciliation

No configuration mismatch was detected between the pinned official training run, final epoch199 checkpoint structure, completed source-A evaluation, and active held-out B invocation.

The checkpoint contains model, EMA1, EMA2, optimizer and epoch, with no saved argument object. The actual training log and pinned code provide the argument evidence. All12 blocks,768 hidden dimensions,200-sample projection and seven conditioning embeddings match. EMA states have matching float32 shapes. Evaluation uses EMA1.

Both evaluations use normalized microvolts/100,16bipolar channels,1000points,six classes, Gaussian scale1, seed0, Heun50 with final Euler and99 conditional calls at CFG1. Requested256 frequency bins resolve to250×4=1000 features. A has68445 expected/reference samples; B has29421 frozen expected samples.

Inference source uses float32 without autocast. The installed Torch defaults report matmul TF32 disabled and cuDNN TF32 enabled; live CUDA flags were not serialized, so this is source/default evidence. Training used BF16 autocast.

B completion and final archive audits remain pending. This record does not establish published-score comparability. Source, checkpoint, evidence and auditor hashes are retained in reconciliation.json.
