"""Append-friendly HDF5 session writer for data_record."""

from __future__ import annotations

import os
import time
from typing import Optional

import h5py
import numpy as np


class H5SessionWriter:
    """Growing datasets for one recording session."""

    def __init__(
        self,
        path: str,
        *,
        height: int = 480,
        width: int = 640,
        record_rate_hz: float = 20.0,
        ee_ws_url: str = "",
        interface: str = "",
        camera_serial: str = "",
        chunk_frames: int = 32,
    ):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        self._f = h5py.File(path, "w")
        self._n = 0
        self._height = int(height)
        self._width = int(width)

        self._f.attrs["record_rate_hz"] = float(record_rate_hz)
        self._f.attrs["ee_ws_url"] = str(ee_ws_url)
        self._f.attrs["interface"] = str(interface)
        self._f.attrs["camera_serial"] = str(camera_serial)
        self._f.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        chunk = max(1, int(chunk_frames))
        self._timestamp = self._f.create_dataset(
            "timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(chunk,),
        )
        self._rgb = self._f.create_dataset(
            "rgb",
            shape=(0, self._height, self._width, 3),
            maxshape=(None, self._height, self._width, 3),
            dtype=np.uint8,
            chunks=(1, self._height, self._width, 3),
            compression="lzf",
        )
        self._ee = self._f.create_dataset(
            "ee_goal_local_cart",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float64,
            chunks=(chunk, 3),
        )
        self._ee_age = self._f.create_dataset(
            "ee_goal_age_s",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(chunk,),
        )
        self._vel = self._f.create_dataset(
            "base_lin_vel_local",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float64,
            chunks=(chunk, 3),
        )
        self._quat = self._f.create_dataset(
            "base_quat",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.float64,
            chunks=(chunk, 4),
        )

    @property
    def num_frames(self) -> int:
        return self._n

    def append(
        self,
        *,
        timestamp: float,
        rgb: np.ndarray,
        ee_goal_local_cart: np.ndarray,
        ee_goal_age_s: float,
        base_lin_vel_local: np.ndarray,
        base_quat: np.ndarray,
    ) -> None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.shape != (self._height, self._width, 3):
            raise ValueError(
                f"rgb shape {rgb.shape} != ({self._height}, {self._width}, 3)"
            )
        ee = np.asarray(ee_goal_local_cart, dtype=np.float64).reshape(3)
        vel = np.asarray(base_lin_vel_local, dtype=np.float64).reshape(3)
        quat = np.asarray(base_quat, dtype=np.float64).reshape(4)

        i = self._n
        for ds in (
            self._timestamp,
            self._rgb,
            self._ee,
            self._ee_age,
            self._vel,
            self._quat,
        ):
            ds.resize(i + 1, axis=0)

        self._timestamp[i] = float(timestamp)
        self._rgb[i] = rgb
        self._ee[i] = ee
        self._ee_age[i] = float(ee_goal_age_s)
        self._vel[i] = vel
        self._quat[i] = quat
        self._n = i + 1

        if self._n % 50 == 0:
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.attrs["num_frames"] = self._n
            self._f.flush()
            self._f.close()
            self._f = None
            print(f"[H5SessionWriter] closed {self.path} ({self._n} frames)", flush=True)
