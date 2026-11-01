"""Bounded, feedback-checked tools; language models never emit joint angles."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .client import Frame, SimulationClient
from .kinematics import Kinematics


@dataclass(frozen=True)
class PickupGoal:
    object_name: str
    minimum_lift_m: float


class Controller:
    def __init__(self, client: SimulationClient) -> None:
        self.client = client
        self.kinematics = Kinematics()
        scene = client.frame().scene
        self.identity = (scene["session_id"], scene["reset_count"])
        self.goal: PickupGoal | None = None

    def goal_info(self) -> dict | None:
        if self.goal is None:
            return None
        return {"object_name": self.goal.object_name, "minimum_lift_m": self.goal.minimum_lift_m}

    def frame(self) -> Frame:
        frame = self.client.frame()
        if (frame.scene["session_id"], frame.scene["reset_count"]) != self.identity:
            self.client.hold()
            raise RuntimeError("Simulator was reset or restarted. Start a new agent run.")
        return frame

    def wait_for(self, predicate: Callable[[Frame], bool], timeout: float = 8.0) -> Frame:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.frame()
            if predicate(frame):
                return frame
            time.sleep(0.04)
        self.client.hold()
        raise TimeoutError("Tool did not reach its target; holding current pose")

    def move(self, side: str, xyz: list[float]) -> dict:
        frame = self.frame()
        waypoints = self.kinematics.path(frame.state, frame.scene, side, xyz)
        for waypoint in waypoints:
            initial = dict(self.frame().state)
            duration = max(
                0.12,
                max(
                    abs(value - initial[key]) / (100.0 if key == "height.pos" else 0.6)
                    for key, value in waypoint.items()
                ),
            )
            start = time.monotonic()
            while True:
                self.frame()  # Stops progress on reset or stale observations.
                fraction = min((time.monotonic() - start) / duration, 1.0)
                action = dict(initial)
                action.update(
                    {key: initial[key] + fraction * (value - initial[key]) for key, value in waypoint.items()}
                )
                self.client.set_target(action)
                if fraction >= 1:
                    break
                time.sleep(0.05)
            self.wait_for(
                lambda f, expected=waypoint: all(
                    abs(f.state[key] - value) < (4.0 if key == "height.pos" else 0.025)
                    for key, value in expected.items()
                )
            )
        reached = self.wait_for(lambda f: np.linalg.norm(np.array(f.scene["tcp"][side]) - xyz) < 0.015)
        return {"tcp": reached.scene["tcp"][side], "objects": reached.scene["objects"]}

    def execute(self, name: str, arguments: dict) -> dict:
        frame = self.frame()
        if name == "observe":
            return frame.scene
        if name == "cannot_complete":
            reason = arguments.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Provide a reason for not completing this task")
            self.client.hold()
            return {"success": False, "reason": reason}
        if name == "select_target":
            object_name = arguments.get("object_name")
            if not isinstance(object_name, str) or object_name not in frame.scene["objects"]:
                raise ValueError("Choose an object_name from the current scene")
            minimum = float(arguments.get("minimum_lift_m", 0.10))
            if not np.isfinite(minimum) or not 0.05 <= minimum <= 0.25:
                raise ValueError("minimum_lift_m must be between 0.05 and 0.25 metres")
            goal = PickupGoal(object_name, minimum)
            if self.goal is not None and self.goal != goal:
                raise ValueError(
                    "The selected goal is locked for this run; cannot change the target or threshold"
                )
            self.goal = goal
            return {"goal": self.goal_info(), "object": frame.scene["objects"][object_name]}
        if self.goal is None:
            raise ValueError("Call select_target to bind the user's task to a scene object before moving")
        if name == "done":
            return {"success": self.success(), "goal": self.goal_info(), "objects": frame.scene["objects"]}
        side = arguments.get("arm")
        if side not in ("left", "right"):
            raise ValueError("arm must be left or right")
        if name == "set_gripper":
            opening = arguments.get("opening")
            if opening not in ("open", "closed"):
                raise ValueError("opening must be open or closed")
            action = dict(frame.state)
            key = f"{side}_gripper.pos"
            action[key] = -4.5 if opening == "open" else 0.0
            self.client.set_target(action)
            self.wait_for(lambda f: abs(f.state[key] - action[key]) < 0.1)
            return self.frame().scene
        if name == "move_to_object":
            object_name = arguments.get("object_name")
            if object_name != self.goal.object_name:
                raise ValueError(
                    f"This run targets {self.goal.object_name}; cannot approach a different object"
                )
            clearance = float(arguments.get("clearance_m", 0.12))
            if not np.isfinite(clearance) or not 0 <= clearance <= 0.25:
                raise ValueError("clearance_m must be between 0 and 0.25 metres")
            xyz = list(frame.scene["objects"][object_name]["position"])
            xyz[2] += clearance
            return self.move(side, xyz)
        if name == "lift":
            height = float(arguments.get("height_m", 0.15))
            if not np.isfinite(height) or not 0.05 <= height <= 0.25:
                raise ValueError("height_m must be between 0.05 and 0.25 metres")
            if frame.scene["objects"][self.goal.object_name]["held_by"] != side:
                raise ValueError(
                    "This arm is not holding the selected object. Open, approach, then close first."
                )
            xyz = list(frame.scene["tcp"][side])
            xyz[2] += height
            return self.move(side, xyz)
        raise ValueError(f"Unsupported tool: {name}")

    def success(self) -> bool:
        if self.goal is None:
            return False
        obj = self.frame().scene["objects"].get(self.goal.object_name)
        return bool(
            obj
            and obj["held_by"] in ("left", "right")
            and obj["position"][2] - obj["initial_position"][2] >= self.goal.minimum_lift_m
        )


DEMO_PLAN = [
    ("select_target", {"object_name": "cube_blue", "minimum_lift_m": 0.10}),
    ("set_gripper", {"arm": "left", "opening": "open"}),
    ("move_to_object", {"arm": "left", "object_name": "cube_blue", "clearance_m": 0.12}),
    ("move_to_object", {"arm": "left", "object_name": "cube_blue", "clearance_m": 0.0}),
    ("set_gripper", {"arm": "left", "opening": "closed"}),
    ("lift", {"arm": "left", "height_m": 0.15}),
    ("done", {}),
]
