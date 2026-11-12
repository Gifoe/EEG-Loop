"""Reconcile completed official R0 training using logs and process memory samples."""
import json
from pathlib import Path
import re
import sys
import hashlib

root=Path(sys.argv[1])
text=(root/"evidence/baseline-seed0.log").read_text()
completed=[int(x) for x in re.findall(r"Epoch: \[(\d+)\] Total time:",text)]
final_steps=[(int(e),int(n)) for e,n in re.findall(r"Epoch: \[(\d+)\]  \[266/(\d+)\]",text)]
averages=[float(x) for x in re.findall(r"Averaged stats:.*?loss: [^ ]+ \(([^)]+)\)",text)]
assert completed==list(range(200))
assert final_steps==[(i,267) for i in range(200)]
assert len(averages)==200 and all(x==x and abs(x)<float("inf") for x in averages)
assert "Exit status: 0" in text and "Traceback" not in text
pid=21560
memory=[]
for line in (root/"evidence/baseline-gpu-processes.csv").read_text().splitlines():
    fields=line.split(",")
    if len(fields)==3 and fields[1].strip()==str(pid):
        memory.append(int(fields[2].strip()))
allocated=[int(x) for x in re.findall(r"max mem: (\d+)",text)]
epoch_times=re.findall(r"Epoch: \[\d+\] Total time: (\d+):(\d+):(\d+)",text)
durations=[int(h)*3600+int(m)*60+int(s) for h,m,s in epoch_times]
record={
    "upstream_commit":"07f9e6491796f4f2c717d6259b1a2e24afce6a77",
    "training_seed":0,"completed_epochs":len(completed),"epochs_contiguous_zero_based":True,
    "steps_per_epoch":267,"total_optimizer_updates":200*267,
    "effective_batch_size":256,"sample_draws_after_drop_last":200*267*256,
    "finite_epoch_average_losses":True,"epoch_average_loss_first":averages[0],
    "epoch_average_loss_final":averages[-1],"epoch_average_losses":averages,
    "training_timer_hms":re.search(r"Training time: ([^\n]+)",text)[1],
    "command_wall_hms":re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([^\n]+)",text)[1],
    "peak_logged_torch_allocated_mib":max(allocated),
    "training_process_pid":pid,"peak_sampled_process_nvml_mib":max(memory),
    "process_memory_sampling_interval_s":5,"process_memory_samples":len(memory),
    "peak_rss_kib":int(re.search(r"Maximum resident set size \(kbytes\): (\d+)",text)[1]),
    "epoch_training_seconds_min":min(durations),"epoch_training_seconds_max":max(durations),
    "exit_status":0,"checkpoint_bytes":(root/"baseline-seed0/checkpoint-last.pth").stat().st_size,
    "training_done":(root/"baseline-seed0/training.done").is_file(),
    "training_log_sha256":hashlib.sha256(text.encode()).hexdigest(),
    "qualification":"NVML process peak sampled every five seconds, while allocator peak is logged by PyTorch. Device total includes unrelated work and is excluded."
}
assert record["training_done"]
output=root/"evidence/training-completion-r0.json"
output.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({k:v for k,v in record.items() if k!="epoch_average_losses"},indent=2))
