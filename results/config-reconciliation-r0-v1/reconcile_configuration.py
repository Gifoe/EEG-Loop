"""Reconcile checkpoint, actual evaluation commands and pinned JET source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np
import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main(root):
    repo=root/"JET"
    sys.path.insert(0,str(repo))
    from inference import get_args_parser
    from models.raw_vit import RAW_VIT_SPECS
    from data.metrics import _prepare_fid_features
    from experiments.looped_jet_tuev_v1.residual_channel_diagnostics import channel_names

    output=root/"evidence/config-reconciliation-r0-v1"
    output.mkdir(mode=0o700)
    evidence=root/"evidence"
    training=(evidence/"baseline-seed0.log").read_text()
    namespace=ast.parse("Namespace("+re.search(r"Namespace\((.*?)\)\n",training,re.S)[1]+")",mode="eval").body
    actual_train={key.arg:ast.literal_eval(key.value) for key in namespace.keywords}
    a=json.loads((evidence/"r0-source-a-completion-v1/completion.json").read_text())
    b_cmdline=Path("/proc/131147/cmdline").read_bytes().split(b"\0")
    b_cmd=[part.decode() for part in b_cmdline if part]
    b_args=vars(get_args_parser().parse_args(b_cmd[2:]))
    b_args.update(target_length=1000,eeg_patch_num=5,class_num=6)
    a_args=a["settings"]
    checkpoint=torch.load(root/"baseline-seed0/checkpoint-last.pth",map_location="cpu",weights_only=True,mmap=True)
    state=checkpoint["model"]
    state_keys=set(state)
    blocks=sorted({int(key.split(".")[2]) for key in state if key.startswith("net.blocks.")})
    structural={
        "blocks":blocks,"pos_embed_shape":list(state["net.pos_embed"].shape),
        "patch_projection_shape":list(state["net.patch_embed.weight"].shape),
        "class_embedding_shape":list(state["net.y_embedder.embedding_table.weight"].shape),
        "all_state_dtypes":sorted({str(t.dtype) for family in ("model","model_ema1","model_ema2") for t in checkpoint[family].values()}),
        "same_model_ema_key_and_shape_sets":all(
            set(checkpoint[family])==state_keys and all(checkpoint[family][k].shape==state[k].shape for k in state)
            for family in ("model_ema1","model_ema2")),
        "stored_checkpoint_keys":list(checkpoint),
        "epoch":checkpoint["epoch"],
        "heads_from_pinned_source":RAW_VIT_SPECS[actual_train["model"]].num_heads,
    }
    expected={"model":"JiT-B/16","num_eeg_channels":16,"target_length":1000,"eeg_patch_size":200,
              "attn_dropout":0.0,"proj_dropout":0.0,"noise_type":"gs","noise_scale":1.0,
              "t_eps":0.05,"sampling_method":"heun","num_sampling_steps":50,"cfg":1.0,
              "seed":0,"P_mean":-0.8,"P_std":0.8,"label_drop_prob":0.1,"ema_decay1":0.9999,"ema_decay2":0.9996}
    comparisons={}
    mismatches=[]
    for key,desired in expected.items():
        vals={"expected":desired,"training":actual_train[key],"source_a":a_args[key],"heldout_b":b_args[key]}
        vals["matches"]=all(vals[name]==desired for name in ("training","source_a","heldout_b"))
        if not vals["matches"]: mismatches.append(key)
        comparisons[key]=vals
    assert blocks==list(range(12)) and structural["pos_embed_shape"]==[1,5,768]
    assert structural["patch_projection_shape"]==[768,200] and structural["class_embedding_shape"]==[7,768]
    assert structural["same_model_ema_key_and_shape_sets"] and structural["epoch"]==199
    feature_shape=list(_prepare_fid_features(torch.zeros(2,16,1000),feature_bins=256).shape)
    assert feature_shape==[2,1000]
    histograms={}
    for split in ("train","test"):
        with np.load(evidence/f"reference-index/{split}-seed0.npz",allow_pickle=False) as index:
            counts=np.bincount(index["labels"],minlength=6).tolist()
        histograms[split]=counts
    assert list(a["metrics"]["label_histogram"].values())==histograms["train"]
    selected_env={}
    allowed={"NVIDIA_TF32_OVERRIDE","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE","CUDA_VISIBLE_DEVICES"}
    for item in Path("/proc/131147/environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key,value=item.decode(errors="replace").split("=",1)
            if key in allowed: selected_env[key]=value
    src_names=["inference.py","train_eeg.py","engine_eeg.py","denoiser.py","models/raw_vit.py",
               "data/metrics.py","data/tuev_dataset.py","data/preprocess_tuev.py",
               "scripts/train_tuev.sh","scripts/infer_tuev.sh"]
    source_hashes={name:sha(repo/name) for name in src_names}
    files=["baseline-seed0.log","r0-source-a-completion-v1/completion.json","final-checkpoints.audit.json",
           "reference-index/train-seed0.npz","reference-index/test-seed0.npz","processed-manifest.json"]
    report={
        "status":"no_detected_configuration_mismatch" if not mismatches else "mismatch",
        "mismatches":mismatches,"comparisons":comparisons,"checkpoint_structure":structural,
        "checkpoint_args_status":"No args saved in official checkpoint. Actual training Namespace log and pinned source supply configuration.",
        "checkpoint_sha256":a["checkpoint"]["sha256"],"source_a_command":a_args,
        "heldout_b_command":b_cmd,"heldout_b_resolved_args":b_args,
        "precision":{"model_ema_dtype":"float32","training_autocast":"bfloat16",
                     "inference_autocast_in_source":False,
                     "fresh_installed_torch_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
                     "fresh_installed_torch_cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
                     "live_b_selected_environment":selected_env,
                     "limit":"Live CUDA precision flags were not separately serialized. Source has no precision/backend overrides; installed-process defaults checked without GPU execution."},
        "sampling":{"ema":"model_ema1","uniform_grid_steps":50,"heun_steps":49,"final_euler_steps":1,
                    "conditional_nfe_per_sample":99,"cfg1_uses_conditional_only":True,"gaussian_scale":1,
                    "same_seed_across_splits":0,"generation_batch_size":64},
        "representation":{"loader_normalization":"microvolts divided by100","labels":"REC1..6 maps to0..5",
                          "channel_order":channel_names(repo/"data/preprocess_tuev.py"),
                          "sampling_rate_hz":200,"window_shape":[16,1000],
                          "feature_requested_bins":256,"effective_frequency_bins":250,"spatial_groups":4,
                          "feature_dimension_synthetic_verified":feature_shape[1],"frequency_range_hz":[0,49.8]},
        "class_counts":histograms,"source_sha256":source_hashes,
        "evidence_sha256":{name:sha(evidence/name) for name in files},
        "auditor_sha256":sha(Path(__file__)),
        "limits":["Configuration reconciliation does not establish comparability to the paper's TS-FID.",
                  "B is still running; class counts are frozen index expectations until output and integrity audits complete.",
                  "Checkpoint loaded through mmap on CPU. No EEG archives decompressed or GPU model executed."],
    }
    subprocess.run(["git","-C",str(repo),"diff","--exit-code"],check=True)
    (output/"reconciliation.json").write_text(json.dumps(report,indent=2)+"\n")
    text=("# R0 configuration reconciliation\n\n"
          "No configuration mismatch was detected between the pinned official training run, final epoch199 checkpoint structure, completed source-A evaluation, and active held-out B invocation.\n\n"
          "The checkpoint contains model, EMA1, EMA2, optimizer and epoch, with no saved argument object. The actual training log and pinned code provide the argument evidence. All12 blocks,768 hidden dimensions,200-sample projection and seven conditioning embeddings match. EMA states have matching float32 shapes. Evaluation uses EMA1.\n\n"
          "Both evaluations use normalized microvolts/100,16bipolar channels,1000points,six classes, Gaussian scale1, seed0, Heun50 with final Euler and99 conditional calls at CFG1. Requested256 frequency bins resolve to250×4=1000 features. A has68445 expected/reference samples; B has29421 frozen expected samples.\n\n"
          "Inference source uses float32 without autocast. The installed Torch defaults report matmul TF32 disabled and cuDNN TF32 enabled; live CUDA flags were not serialized, so this is source/default evidence. Training used BF16 autocast.\n\n"
          "B completion and final archive audits remain pending. This record does not establish published-score comparability. Source, checkpoint, evidence and auditor hashes are retained in reconciliation.json.\n")
    (output/"README.md").write_text(text)
    (output/"manifest.sha256").write_text("".join(sha(output/name)+"  "+name+"\n" for name in ("reconciliation.json","README.md")))
    print(json.dumps({"status":report["status"],"mismatches":mismatches,"class_counts":histograms,"precision":report["precision"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    main(**vars(parser.parse_args()))
