"""Phase budgets, counting existing owned checkpoints once per inode."""
import shutil

SAFETY=5*1024**3
GENERATED_UPPER=int(6263424000*1.02)+20_000_000


def occupied_checkpoints(output):
    unique={}
    for path in output.glob("checkpoint-*.pth"):
        if path.is_symlink():
            raise ValueError("Checkpoint symlink")
        info=path.stat()
        unique[info.st_dev,info.st_ino]=info.st_size
    return sum(unique.values())


def required(phase,checkpoint_upper,occupied=0,view_upper=0):
    if phase=="materialization":
        return view_upper+SAFETY
    if phase=="training":
        return max(0,4*checkpoint_upper-occupied)+SAFETY
    if phase=="evaluation":
        return max(0,3*checkpoint_upper-occupied)+GENERATED_UPPER+SAFETY
    raise ValueError("Unknown storage phase")


def check(root,phase,checkpoint_upper,occupied=0,view_upper=0):
    minimum=required(phase,checkpoint_upper,occupied,view_upper)
    free=shutil.disk_usage(root).free
    if free<minimum:
        raise ValueError(f"Insufficient {phase} storage: free={free}, required={minimum}")
    return {"phase":phase,"free_bytes":free,"required_free_bytes":minimum,
            "existing_unique_checkpoint_bytes":occupied,"safety_bytes":SAFETY}
