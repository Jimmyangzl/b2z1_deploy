#!/usr/bin/env python3
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from paths import setup_paths

setup_paths()

from policy.constants import num_actions, num_observations
from wbc_runtime import ProcessPolicyWorker


def main():
    ckpt = os.path.join(ROOT, "wbc_ckpt/b2z1-wbc-deploy-small-step/model_32000.pt")
    w = ProcessPolicyWorker(checkpoint=ckpt, device="cpu", num_threads=1)
    w.start()
    time.sleep(4.0)
    obs = np.random.randn(num_observations).astype(np.float64)
    got = None
    for _ in range(80):
        w.submit(obs)
        time.sleep(0.02)
        a = w.take_action()
        if a is not None:
            got = a
    print(
        "infer_count",
        w.infer_count,
        "wall",
        w.last_infer_ms,
        "cpu",
        w.last_cpu_ms,
        "wait",
        w.last_wait_ms,
    )
    print("action", None if got is None else got.shape)
    assert got is not None and got.shape == (num_actions,)
    assert w.infer_count > 0
    assert w.last_infer_ms < 50.0, w.last_infer_ms
    w.stop()
    print("OK")


if __name__ == "__main__":
    main()
