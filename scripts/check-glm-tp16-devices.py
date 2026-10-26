#!/usr/bin/env python3
"""Check every physical device in a fresh process before full TP16 loading."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    code = """
import json,torch,torch_npu
torch.npu.set_device(0)
x=torch.ones(8,device='npu')
assert x.cpu().sum().item()==8
free,total=torch.npu.mem_get_info()
print(json.dumps({'free_bytes':free,'total_bytes':total,'graph':False}))
"""
    records = []
    for device in range(16):
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(device))
        try:
            p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                               text=True, timeout=45)
            row = {"device": device, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
            if p.returncode == 0:
                row.update(json.loads(p.stdout.strip().splitlines()[-1]))
        except subprocess.TimeoutExpired as exc:
            row = {"device": device, "returncode": "timeout", "error": str(exc)}
        records.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("stdout", "stderr")}), flush=True)
        args.output.write_text(json.dumps({"complete": len(records) == 16, "devices": records,
                                           "updated_at_unix": time.time()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
