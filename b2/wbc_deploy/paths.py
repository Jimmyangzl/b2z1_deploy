"""Add wbc_deploy, SDK, and third-party paths to sys.path."""

import os
import sys

WBC_DEPLOY_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(WBC_DEPLOY_ROOT, "..", ".."))
Z1_DIR = os.path.join(REPO_ROOT, "z1")
B2_SDK_DIR = os.path.join(REPO_ROOT, "b2", "unitree_sdk2_python")
RSL_RL_DIR = os.path.join(WBC_DEPLOY_ROOT, "third_party", "rsl_rl")
Z1_SDK_LIB = os.path.join(REPO_ROOT, "z1", "z1_sdk", "lib")
B2_CONST_DIR = os.path.join(REPO_ROOT, "b2", "unitree_sdk2_python", "example", "b2", "low_level")
SHIMS_DIR = os.path.join(WBC_DEPLOY_ROOT, "shims")


def setup_paths() -> None:
    for path in (WBC_DEPLOY_ROOT, SHIMS_DIR, Z1_DIR, B2_SDK_DIR, RSL_RL_DIR, Z1_SDK_LIB, B2_CONST_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
