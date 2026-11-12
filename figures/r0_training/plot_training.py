"""Extract verified R0 epoch records, or render the supplied aggregate CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(log, completion, output):
    text=log.read_text()
    sequence=[int(x) for x in re.findall(r"Epoch: \[(\d+)\] Total time:",text)]
    loss=[float(x) for x in re.findall(r"Averaged stats:.*?loss: [^ ]+ \(([^)]+)\)",text)]
    durations=[int(h)*3600+int(m)*60+int(s) for h,m,s in re.findall(
        r"Epoch: \[\d+\] Total time: (\d+):(\d+):(\d+)",text)]
    verified=json.loads(completion.read_text())
    assert sequence==list(range(200)) and len(loss)==len(durations)==200
    assert loss==verified["epoch_average_losses"]
    assert digest(log)==verified["training_log_sha256"]
    output.mkdir(parents=True,exist_ok=True)
    with (output/"epochs.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("epoch_zero_based","epoch_human","epoch_mean_training_objective","observed_epoch_seconds"))
        writer.writerows(zip(sequence,range(1,201),loss,durations,strict=True))
    meta={key:verified[key] for key in ("upstream_commit","training_seed","training_log_sha256",
         "training_timer_hms","command_wall_hms","peak_logged_torch_allocated_mib")}
    meta.update(epochs=200,device_context="Shared NVIDIA RTX A6000",
                duration_definition="Original Total time per epoch rounded to integer seconds. Checkpoint writes and setup excluded.",
                objective_definition="Original logged raw epoch-mean mixed training objective rounded to four decimals.")
    (output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")


def render(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data=np.genfromtxt(output/"epochs.csv",delimiter=",",names=True)
    meta=json.loads((output/"metadata.json").read_text())
    assert np.array_equal(data["epoch_human"],np.arange(1,201))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":1.1,
        "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5))
    axes[0].plot(data["epoch_human"],data["epoch_mean_training_objective"],color="#0F4D92",linewidth=1.6)
    axes[1].plot(data["epoch_human"],data["observed_epoch_seconds"],color="#4D4D4D",linewidth=1.4)
    axes[0].set(title="a   Raw epoch-mean training objective",ylabel="Training objective",ylim=(0,1.32))
    axes[1].set(title="b   Observed epoch duration",ylabel="Seconds / epoch",ylim=(0,230))
    for ax in axes:
        ax.set(xlabel="Epoch",xlim=(1,200),xticks=[1,50,100,150,200])
        ax.tick_params(direction="out",width=1)
        ax.grid(False)
    fig.suptitle("JET on TUEV · seed 0 · shared RTX A6000",fontsize=13,y=0.98)
    fig.text(0.065,0.078,"200 original logged epoch means; durations exclude checkpoint writes and setup.",fontsize=9,color="#4D4D4D")
    fig.text(0.065,0.027,"Training log SHA-256  "+meta["training_log_sha256"],fontsize=6.8,color="#4D4D4D")
    fig.tight_layout(rect=(0.02,0.13,1,0.93),pad=1.2,w_pad=2)
    for suffix in ("png","svg","pdf"):
        fig.savefig(output/f"r0_training.{suffix}",dpi=300,facecolor="white")
    plt.close(fig)
    hashes={path.name:digest(path) for path in output.iterdir()
            if path.is_file() and path.suffix in (".csv",".json",".png",".svg",".pdf",".py",".md") and path.name!="artifact-manifest.json"}
    (output/"artifact-manifest.json").write_text(json.dumps({
        "files_sha256":hashes,"original_log_sha256":meta["training_log_sha256"],
        "matplotlib_version":matplotlib.__version__,"font":"DejaVu Sans",
        "exports":{"png_dpi":300,"svg_editable_text":True,"pdf_fonttype":42},
    },indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--log",type=Path)
    parser.add_argument("--completion",type=Path)
    args=parser.parse_args()
    if args.log:
        extract(args.log,args.completion,args.output)
    else:
        render(args.output)

