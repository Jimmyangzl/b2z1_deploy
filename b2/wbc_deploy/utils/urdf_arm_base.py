"""Read Z1 arm mount offset from b2z1 URDF (base_static_joint)."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List


def load_arm_base_offset_from_urdf(urdf_path: str) -> List[float]:
    """Return xyz of fixed joint ``base_static_joint`` (meters in robot base frame).

    Uses ElementTree so URDF arithmetic expressions do not break parsing.
    """
    path = os.path.abspath(os.path.expanduser(urdf_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"URDF not found: {path}")

    root = ET.parse(path).getroot()
    for joint in root.findall("joint"):
        if joint.get("name") != "base_static_joint":
            continue
        origin = joint.find("origin")
        if origin is None or origin.get("xyz") is None:
            raise ValueError(f"base_static_joint has no origin xyz in URDF: {path}")
        xyz = [float(x) for x in origin.get("xyz").split()]
        if len(xyz) != 3:
            raise ValueError(f"base_static_joint xyz must have 3 values, got {xyz} in {path}")
        return xyz
    raise ValueError(f"base_static_joint not found in URDF: {path}")
