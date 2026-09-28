"""Non-blocking keyboard base-velocity commands for WBC deploy."""

from __future__ import annotations

import select
import sys
import termios
import threading
import tty
from typing import Callable, Optional


class VelKeyboardController:
    """Update base velocity commands from single-key presses (Linux tty).

    Keys (defaults; magnitudes come from config ``keyboard_vel``):
      w/s  → ±vx
      a/d  → ±vy
      q/e  → ±yaw
      space → zero all
    """

    def __init__(
        self,
        keyboard_cfg: dict,
        on_update: Optional[Callable[[float, float, float], None]] = None,
    ):
        kb = keyboard_cfg or {}
        self.vx_fwd = float(kb.get("vx_fwd", 0.4))
        self.vx_back = float(kb.get("vx_back", -0.4))
        self.vy_left = float(kb.get("vy_left", 0.3))
        self.vy_right = float(kb.get("vy_right", -0.3))
        self.yaw_left = float(kb.get("yaw_left", 0.3))
        self.yaw_right = float(kb.get("yaw_right", -0.3))
        self.on_update = on_update
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_term: Optional[list] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if not sys.stdin.isatty():
            print("[vel_keyboard] stdin is not a TTY; keyboard control disabled.")
            return
        self._thread = threading.Thread(target=self._loop, name="vel_keyboard", daemon=True)
        self._thread.start()
        print(
            "[vel_keyboard] active: "
            f"w/s vx=±{abs(self.vx_fwd):.2f}  "
            f"a/d vy=±{abs(self.vy_left):.2f}  "
            f"q/e yaw=±{abs(self.yaw_left):.2f}  "
            "space=stop"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._restore_terminal()

    def _notify(self) -> None:
        if self.on_update is not None:
            self.on_update(self.vx, self.vy, self.yaw)

    def _apply_key(self, ch: str) -> None:
        if ch in ("w", "W"):
            self.vx = self.vx_fwd
        elif ch in ("s", "S"):
            self.vx = self.vx_back
        elif ch in ("a", "A"):
            self.vy = self.vy_left
        elif ch in ("d", "D"):
            self.vy = self.vy_right
        elif ch in ("q", "Q"):
            self.yaw = self.yaw_left
        elif ch in ("e", "E"):
            self.yaw = self.yaw_right
        elif ch == " ":
            self.vx = 0.0
            self.vy = 0.0
            self.yaw = 0.0
        else:
            return
        print(
            f"[vel_keyboard] cmd vx={self.vx:.2f} vy={self.vy:.2f} yaw={self.yaw:.2f}",
            flush=True,
        )
        self._notify()

    def _setup_terminal(self) -> None:
        fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    def _restore_terminal(self) -> None:
        if self._old_term is None:
            return
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term)
        except Exception:
            pass
        self._old_term = None

    def _loop(self) -> None:
        try:
            self._setup_terminal()
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                self._apply_key(ch)
        finally:
            self._restore_terminal()
