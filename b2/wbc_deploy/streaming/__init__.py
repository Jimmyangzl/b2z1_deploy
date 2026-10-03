"""Streaming helpers for remote data recording."""

__all__ = ["EeGoalWsServer"]


def __getattr__(name: str):
    if name == "EeGoalWsServer":
        from streaming.ee_goal_ws_server import EeGoalWsServer

        return EeGoalWsServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
