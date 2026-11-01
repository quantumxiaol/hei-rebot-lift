#!/usr/bin/env python
"""Lightweight ZMQ protocol shared by the MuJoCo and LeRobot environments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = 1
ROBOT_TYPE = "hei_rebot_lift_client"
OBSERVATION_TOPIC = "hei_mujoco_observation"
DEFAULT_OBSERVATION_PORT = 6565
DEFAULT_COMMAND_PORT = 6566
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_NAMES = ("front", "left_wrist", "right_wrist")
ARM_POSITION_NAMES = tuple(
    f"{side}_joint_{joint}.pos"
    for side in ("right", "left")
    for joint in range(1, 7)
)
GRIPPER_POSITION_NAMES = ("right_gripper.pos", "left_gripper.pos")
BASE_AND_LIFT_NAMES = ("x.vel", "y.vel", "theta.vel", "height.pos")
STATE_NAMES = (
    *(f"right_joint_{joint}.pos" for joint in range(1, 7)),
    "right_gripper.pos",
    *(f"left_joint_{joint}.pos" for joint in range(1, 7)),
    "left_gripper.pos",
    *BASE_AND_LIFT_NAMES,
)


def state_feature_types() -> dict[str, type]:
    return {name: float for name in STATE_NAMES}


def observation_feature_types(
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
) -> dict[str, type | tuple[int, int, int]]:
    return {
        **state_feature_types(),
        **{name: (int(height), int(width), 3) for name in CAMERA_NAMES},
    }


def tcp_endpoint(host: str, port: int) -> str:
    host = host.strip()
    if not host:
        raise ValueError("ZMQ host must not be empty")
    return f"tcp://{host}:{int(port)}"


def _numeric_fields(values: Mapping[str, object], label: str) -> dict[str, float]:
    missing = [name for name in STATE_NAMES if name not in values]
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")
    result = {name: float(values[name]) for name in STATE_NAMES}
    if not np.all(np.isfinite(tuple(result.values()))):
        raise ValueError(f"{label} contains non-finite values")
    return result


def encode_observation_packet(
    *,
    topic: str,
    sequence: int,
    timestamp_s: float,
    source_mode: str,
    publish_fps: float,
    observation: Mapping[str, object],
    action: Mapping[str, object],
    jpeg_quality: int = 85,
    simulation: Mapping[str, object] | None = None,
) -> list[bytes]:
    """Encode one synchronized state/action/three-camera sample."""
    import cv2

    if source_mode not in {"vr", "keyboard", "policy"}:
        raise ValueError(f"Unsupported source_mode: {source_mode}")
    quality = int(np.clip(jpeg_quality, 20, 100))
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "sequence": int(sequence),
        "timestamp_s": float(timestamp_s),
        "source_mode": source_mode,
        "publish_fps": float(publish_fps),
        "observation": _numeric_fields(observation, "observation"),
        "action": _numeric_fields(action, "action"),
        "cameras": {
            name: {
                "encoding": "jpeg_rgb8",
                "shape": list(np.asarray(observation[name]).shape),
            }
            for name in CAMERA_NAMES
        },
    }
    if simulation is not None:
        metadata["simulation"] = dict(simulation)
    frames = [topic.encode("utf-8"), json.dumps(metadata, separators=(",", ":")).encode("utf-8")]
    for name in CAMERA_NAMES:
        rgb = np.asarray(observation[name])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Camera {name} must be an HxWx3 uint8 RGB image, got {rgb.shape}/{rgb.dtype}")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            raise RuntimeError(f"Failed to JPEG-encode camera {name}")
        frames.append(encoded.tobytes())
    return frames


def decode_observation_packet(
    parts: Sequence[bytes],
    expected_topic: str = OBSERVATION_TOPIC,
) -> tuple[dict, dict[str, float | np.ndarray], dict[str, float]]:
    """Decode and validate one observation packet."""
    import cv2

    expected_count = 2 + len(CAMERA_NAMES)
    if len(parts) != expected_count:
        raise ValueError(f"Expected {expected_count} multipart frames, received {len(parts)}")
    topic = parts[0].decode("utf-8")
    if topic != expected_topic:
        raise ValueError(f"Unexpected topic {topic!r}, expected {expected_topic!r}")
    metadata = json.loads(parts[1].decode("utf-8"))
    if int(metadata.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol version {metadata.get('protocol_version')}; expected {PROTOCOL_VERSION}"
        )

    observation: dict[str, float | np.ndarray] = _numeric_fields(
        metadata.get("observation", {}),
        "observation",
    )
    action = _numeric_fields(metadata.get("action", {}), "action")
    for name, payload in zip(CAMERA_NAMES, parts[2:], strict=True):
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode camera {name}")
        observation[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return metadata, observation, action


def safe_hold_action(observation: Mapping[str, object]) -> dict[str, float]:
    """Hold arm/lift positions and stop chassis motion."""
    action = _numeric_fields(observation, "observation")
    action["x.vel"] = 0.0
    action["y.vel"] = 0.0
    action["theta.vel"] = 0.0
    return action
