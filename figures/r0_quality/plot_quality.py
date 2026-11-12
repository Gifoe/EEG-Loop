"""Plot measured aggregate seed-0 held-out quality evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(inputs,output):
    output.mkdir(exist_ok=True,parents=True)
    summary=json.loads((inputs/"calibrated-summary.json").read_text())
    residual=json.loads((inputs/"residual-summary.json").read_text())
    families=("psd","band_power","morphology","temporal","spatial")
    labels=("SPSW","GPED","PLED","EYEM","ARTF","BCKG")
    upper=np.triu_indices(16,1)
    values=np.array([[summary["by_class"][str(c)]["family_means"][f] for f in families] for c in range(6)])
    points=np.array([[np.abs(np.asarray(residual["by_class"][str(c)][side]["mean_residual_correlation"])[upper]).mean()
                      for side in ("reference","generated")] for c in range(6)])
    counts=np.array([[residual["by_class"][str(c)][side]["mean_estimation_windows"] for side in ("reference","generated")] for c in range(6)])
    assert np.isfinite(values).all() and np.isfinite(points).all()
    assert np.array_equal(counts[:,0],counts[:,1])
    matched=summary["frozen_plan"]["class_counts"]
    with (output/"source_values.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("class",*families,"reference_mean_absolute_class_mean_residual_correlation",
                         "generated_mean_absolute_class_mean_residual_correlation","residual_windows_per_domain","calibrated_windows_per_domain"))
        for c,label in enumerate(labels):
            writer.writerow((label,*values[c],*points[c],counts[c,0],matched[c]))
    loaded=np.genfromtxt(output/"source_values.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
    assert all(np.array_equal(loaded[f],values[:,i]) for i,f in enumerate(families))
    assert np.array_equal(loaded["reference_mean_absolute_class_mean_residual_correlation"],points[:,0])
    assert np.array_equal(loaded["generated_mean_absolute_class_mean_residual_correlation"],points[:,1])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,"axes.linewidth":1.1,
        "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(11.6,5.2),gridspec_kw={"width_ratios":[1.25,1]})
    im=axes[0].imshow(values,cmap="Blues",vmin=0,vmax=12,aspect="auto")
    axes[0].set(xticks=np.arange(5),xticklabels=["PSD","Band\npower","Morphology","Temporal","Spatial"],
                yticks=np.arange(6),yticklabels=labels,title="a   Calibrated class-family discrepancies")
    for row in range(6):
        for col in range(5):
            axes[0].text(col,row,f"{values[row,col]:.2f}",ha="center",va="center",
                         color="white" if values[row,col]>6 else "#17212A",fontsize=10)
    axes[0].tick_params(length=0,pad=7)
    cb=fig.colorbar(im,ax=axes[0],fraction=0.048,pad=0.04,ticks=[0,3,6,9,12])
    cb.set_label("Discrepancy / frozen real–real scale",fontsize=9)
    x=np.arange(6)
    for c in range(6):
        axes[1].plot([c,c],points[c],color="#CFCECE",lw=1.5,zorder=1)
    axes[1].scatter(x,points[:,0],label="Reference",color="#0F4D92",s=38,zorder=3)
    axes[1].scatter(x,points[:,1],label="Generated",color="#B64342",marker="s",s=31,zorder=3)
    axes[1].set(xticks=x,xticklabels=labels,ylim=(-0.012,0.4),yticks=[0,0.1,0.2,0.3,0.4],xlim=(-0.45,5.45),
                ylabel="Mean |class-mean residual correlation|\nacross 120 channel pairs",
                title="b   Residual channel dependence")
    axes[1].legend(frameon=False,loc="upper right",fontsize=9)
    axes[1].tick_params(axis="x",labelsize=9)
    fig.suptitle("Official JET · seed 0 · held-out TUEV",fontsize=14,y=0.98)
    fig.tight_layout(rect=(0.01,0.20,1,0.94),w_pad=2.6,pad=1.2)
    fig.text(0.07,0.13,"a  Means of 200 overlapping subset draws, normalized by frozen train/validation real–real scales.",fontsize=9)
    fig.text(0.07,0.085,"b  Absolute values are taken after averaging residual correlation matrices within each class and domain.",fontsize=9)
    fig.text(0.07,0.04,"Single-checkpoint descriptive results. Class means use the evaluated samples; finite sample counts affect residual estimates.",fontsize=8.5,color="#4D4D4D")
    for ext in ("png","svg","pdf"):
        fig.savefig(output/f"r0_quality.{ext}",dpi=300,facecolor="white")
    plt.close(fig)
    metadata={"inputs_sha256":{name:sha(inputs/name) for name in ("calibrated-summary.json","residual-summary.json")},
              "training_seed":0,"scope":"held-out TUEV, one reproduced checkpoint",
              "families":families,"class_order":labels,"calibrated_class_counts":matched,
              "full_class_counts":counts[:,0].tolist(),"residual_statistic":"mean over120 upper-triangle entries of absolute class/domain mean residual Pearson matrix",
              "calibrated_statistic":"Equal channel and primitive weighting within family, then mean over200 fixed overlapping subset repetitions.",
              "uncertainty":"Descriptive point summaries only; repetitions do not represent independent training seeds.",
              "font":"DejaVu Sans","matplotlib":matplotlib.__version__}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    hashes={path.name:sha(path) for path in output.iterdir() if path.is_file() and path.name!="manifest.json"}
    (output/"manifest.json").write_text(json.dumps({"files_sha256":hashes},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path)
    parser.add_argument("output",type=Path)
    main(**vars(parser.parse_args()))

