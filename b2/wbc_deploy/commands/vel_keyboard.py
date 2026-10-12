"""Non-blocking terminal hotkeys for WBC deploy (velocity + gripper)."""

from __future__ import annotations

import select
import sys
import termios
import threading
import tty
from typing import Callable, Optional


class DeployHotkeys:
    """Single stdin listener for deploy keyboard controls (Linux tty).

    Keys:
      space → toggle gripper (if ``on_gripper_toggle`` set)
      w/s   → ±vx   (if velocity enabled)
      a/d   → ±vy
      q/e   → ±yaw
      0     → zero vx/vy/yaw
    """

    def __init__(
        self,
        keyboard_cfg: Optional[dict] = None,
        *,
        enable_vel: bool = False,
        on_vel_update: Optional[Callable[[float, float, float], None]] = None,
        on_gripper_toggle: Optional[Callable[[], None]] = None,
    ):
        kb = keyboard_cfg or {}
        self.vx_fwd = float(kb.get("vx_fwd", 0.4))
        self.vx_back = float(kb.get("vx_back", -0.4))
        self.vy_left = float(kb.get("vy_left", 0.3))
        self.vy_right = float(kb.get("vy_right", -0.3))
        self.yaw_left = float(kb.get("yaw_left", 0.3))
        self.yaw_right = float(kb.get("yaw_right", -0.3))
        self.on_vel_update = on_vel_update
        self.on_gripper_toggle = on_gripper_toggle
        self.enable_vel = bool(enable_vel)
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_term: Optional[list] = None
        self._lock = threading.Lock()
        self._paused = False

    def set_velocity_enabled(
        self,
        enabled: bool,
        on_vel_update: Optional[Callable[[float, float, float], None]] = None,
    ) -> None:
        with self._lock:
            self.enable_vel = bool(enabled)
            if on_vel_update is not None:
                self.on_vel_update = on_vel_update
        if enabled:
            print(
                "[hotkeys] velocity: "
                f"w/s vx=±{abs(self.vx_fwd):.2f}  "
                f"a/d vy=±{abs(self.vy_left):.2f}  "
                f"q/e yaw=±{abs(self.yaw_left):.2f}  "
                "0=stop",
                flush=True,
            )

    def start(self) -> None:
        if self._thread is not None:
            return
        if not sys.stdin.isatty():
            print("[hotkeys] stdin is not a TTY; keyboard control disabled.")
            return
        self._thread = threading.Thread(target=self._loop, name="deploy_hotkeys", daemon=True)
        self._thread.start()
        parts = []
        if self.on_gripper_toggle is not None:
            parts.append("space=gripper toggle (closed=0.2 / open=-1)")
        if self.enable_vel:
            parts.append(
                f"w/s vx=±{abs(self.vx_fwd):.2f}  "
                f"a/d vy=±{abs(self.vy_left):.2f}  "
                f"q/e yaw=±{abs(self.yaw_left):.2f}  "
                "0=stop"
            )
        print(f"[hotkeys] active: {';  '.join(parts) if parts else 'none'}", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._restore_terminal()

    def pause_for_input(self) -> None:
        """Stop cbreak listener so cooked ``input()`` can read a full line (e.g. YES)."""
        was_running = self._thread is not None and self._thread.is_alive()
        self.stop()
        self._paused = was_running
        # Drop any leftover keystrokes queued while in cbreak.
        if sys.stdin.isatty():
            try:
                termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            except Exception:
                pass

    def resume_after_input(self) -> None:
        """Restart after ``pause_for_input`` if it was previously running."""
        if not self._paused:
            return
        self._paused = False
        self._stop.clear()
        self.start()

    def _notify_vel(self) -> None:
        cb = self.on_vel_update
        if cb is not None:
            cb(self.vx, self.vy, self.yaw)

    def _apply_key(self, ch: str) -> None:
        if ch == " " and self.on_gripper_toggle is not None:
            self.on_gripper_toggle()
            return

        with self._lock:
            vel_on = self.enable_vel
        if not vel_on:
            return

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
        elif ch == "0":
            self.vx = 0.0
            self.vy = 0.0
            self.yaw = 0.0
        else:
            return
        print(
            f"[hotkeys] cmd vx={self.vx:.2f} vy={self.vy:.2f} yaw={self.yaw:.2f}",
            flush=True,
        )
        self._notify_vel()

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


# Back-compat alias used by older call sites / docs.
class VelKeyboardController(DeployHotkeys):
    """Velocity-focused hotkeys; prefer ``DeployHotkeys`` for new code."""

    def __init__(
        self,
        keyboard_cfg: dict,
        on_update: Optional[Callable[[float, float, float], None]] = None,
    ):
        super().__init__(
            keyboard_cfg,
            enable_vel=True,
            on_vel_update=on_update,
            on_gripper_toggle=None,
        )
