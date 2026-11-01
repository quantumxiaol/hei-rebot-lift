"""One socket-owning thread receives images and maintains the simulation heartbeat."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Self

import zmq

from . import paths  # noqa: F401

# isort: split
from hei_robot_mujoco_zmq_protocol import OBSERVATION_TOPIC, safe_hold_action


@dataclass(frozen=True)
class Frame:
    metadata: dict
    images: tuple[bytes, ...]
    received_s: float

    @property
    def state(self) -> dict[str, float]:
        return self.metadata["observation"]

    @property
    def scene(self) -> dict:
        return self.metadata["simulation"]


class SimulationClient:
    """Local simulation only. Never connects to real-robot ports or serial devices."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame: Frame | None = None
        self._target: dict[str, float] | None = None
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="sim-agent-io", daemon=True)
        self._thread.start()

    def frame(self, timeout: float = 5.0) -> Frame:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._error:
                    raise RuntimeError(f"Simulation connection failed: {self._error}") from self._error
                if self._frame and time.monotonic() - self._frame.received_s < 1.5:
                    return self._frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "No fresh simulator observation on 127.0.0.1:6565. Start the simulator first."
                    )
                self._condition.wait(min(remaining, 0.1))

    def set_target(self, action: dict[str, float]) -> None:
        target = safe_hold_action(action)
        with self._condition:
            self._target = target

    def hold(self) -> None:
        with self._condition:
            self._target = safe_hold_action(self._frame.state) if self._frame else None

    def _run(self) -> None:
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        sender = context.socket(zmq.PUSH)
        verified = False
        try:
            subscriber.setsockopt(zmq.SUBSCRIBE, OBSERVATION_TOPIC.encode())
            subscriber.setsockopt(zmq.RCVHWM, 2)  # CONFLATE breaks multipart images.
            subscriber.setsockopt(zmq.LINGER, 0)
            sender.setsockopt(zmq.SNDHWM, 2)
            sender.setsockopt(zmq.IMMEDIATE, 1)
            sender.setsockopt(zmq.LINGER, 0)
            subscriber.connect("tcp://127.0.0.1:6565")
            sender.connect("tcp://127.0.0.1:6566")
            next_send = time.monotonic()
            while not self._stop.is_set():
                if subscriber.poll(20):
                    parts = subscriber.recv_multipart()
                    for _ in range(8):
                        try:
                            parts = subscriber.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                    metadata = json.loads(parts[1])
                    if len(parts) != 5 or metadata.get("protocol_version") != 1:
                        raise ValueError("Unsupported simulation observation protocol")
                    scene = metadata.get("simulation", {})
                    if scene.get("kind") != "hei_mujoco_kinematic" or metadata.get("source_mode") != "policy":
                        raise ValueError("Requires the updated local MuJoCo server in policy mode")
                    safe_hold_action(metadata["observation"])
                    if abs(time.time() - float(metadata["timestamp_s"])) > 1.5:
                        continue
                    verified = True
                    frame = Frame(metadata, tuple(parts[2:]), time.monotonic())
                    with self._condition:
                        if self._frame and (scene["session_id"], scene["reset_count"]) != (
                            self._frame.scene["session_id"],
                            self._frame.scene["reset_count"],
                        ):
                            self._target = None
                            raise RuntimeError("Simulator reset or restarted; start a new agent run")
                        self._frame = frame
                        self._condition.notify_all()
                now = time.monotonic()
                if now < next_send:
                    continue
                next_send = now + 0.1
                with self._condition:
                    fresh = self._frame and now - self._frame.received_s < 1.5
                    if not fresh:
                        self._target = None
                    action = dict(self._target or safe_hold_action(self._frame.state)) if fresh else None
                if action is not None:
                    try:
                        sender.send_json({"type": "policy_action", "action": action}, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass
        except Exception as exc:  # noqa: BLE001 — propagate worker failures to the control thread.
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            if verified:
                try:
                    sender.send_json({"type": "stop"}, flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    pass
            subscriber.close(0)
            sender.close(0)
            context.term()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("Simulation IO thread did not stop")

    def __enter__(self) -> Self:
        try:
            self.frame()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, *_args) -> None:
        self.close()
