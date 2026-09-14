import os

import torch

lr = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
print(
    f"  task lr={lr} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"device_count={torch.cuda.device_count()}",
    flush=True,
)
try:
    torch.cuda.set_device(lr)
    x = torch.zeros(8, device=f"cuda:{lr}")
    print(f"  task lr={lr}: cuda:{lr} OK", flush=True)
except Exception as e:
    print(f"  task lr={lr}: FAILED -> {type(e).__name__}: {str(e)[:90]}", flush=True)
