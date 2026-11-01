"""MuJoCo damped least-squares IK: six arm joints plus the shared lift."""

from __future__ import annotations

import mujoco
import numpy as np

from . import paths  # noqa: F401

# isort: split
from hei_robot_mujoco_scene import build_mujoco_model
from hei_robot_vr_mujoco_sim import ARM_JOINTS, BASE_FRAME, DEFAULT_URDF, LIFT_JOINT, TCP_FRAMES


class Kinematics:
    def __init__(self) -> None:
        self.model = build_mujoco_model(DEFAULT_URDF, add_environment=False)
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body(BASE_FRAME).id
        self.base_position = self.model.body_pos[self.base_id].copy()
        self.base_quaternion = self.model.body_quat[self.base_id].copy()

    def load(self, state: dict[str, float], scene: dict) -> None:
        self.data.qpos[:] = 0
        for side, names in ARM_JOINTS.items():
            for index, name in enumerate(names, 1):
                self.data.qpos[self.model.joint(name).qposadr[0]] = state[f"{side}_joint_{index}.pos"]
        self.data.qpos[self.model.joint(LIFT_JOINT).qposadr[0]] = state["height.pos"] / 1000
        x, y, yaw = scene["base_pose"]
        self.model.body_pos[self.base_id] = self.base_position + np.array([x, y, 0.0])
        rotation = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        mujoco.mju_mulQuat(self.model.body_quat[self.base_id], rotation, self.base_quaternion)
        mujoco.mj_forward(self.model, self.data)

    def solve(self, side: str, xyz: np.ndarray) -> dict[str, float]:
        if side not in ARM_JOINTS:
            raise ValueError("arm must be left or right")
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("TCP target must be three finite numbers")
        if not (0.10 <= xyz[0] <= 1.10 and abs(xyz[1]) <= 0.70 and 0.20 <= xyz[2] <= 1.60):
            raise ValueError("TCP target is outside the demonstration workspace")
        joints = [self.model.joint(name) for name in (*ARM_JOINTS[side], LIFT_JOINT)]
        q_indices = np.array([joint.qposadr[0] for joint in joints])
        dof_indices = np.array([joint.dofadr[0] for joint in joints])
        limits = np.array([joint.range for joint in joints])
        tcp_id = self.model.body(TCP_FRAMES[side]).id
        jacobian = np.zeros((3, self.model.nv))
        for _ in range(400):
            mujoco.mj_forward(self.model, self.data)
            error = xyz - self.data.xpos[tcp_id]
            if np.linalg.norm(error) < 0.003:
                break
            mujoco.mj_jacBody(self.model, self.data, jacobian, None, tcp_id)
            jac = jacobian[:, dof_indices]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.0004 * np.eye(3), error)
            step_limit = np.array([0.08] * 6 + [0.015])
            q = self.data.qpos[q_indices] + np.clip(delta, -step_limit, step_limit)
            self.data.qpos[q_indices] = np.clip(q, limits[:, 0], limits[:, 1])
        mujoco.mj_forward(self.model, self.data)
        distance = float(np.linalg.norm(xyz - self.data.xpos[tcp_id]))
        if distance > 0.008:
            raise ValueError(f"Unreachable {side} TCP target {xyz.tolist()}: residual {distance:.3f} m")
        result = {f"{side}_joint_{i + 1}.pos": float(self.data.qpos[q_indices[i]]) for i in range(6)}
        result["height.pos"] = float(self.data.qpos[q_indices[-1]] * 1000)
        return result

    def path(self, state: dict[str, float], scene: dict, side: str, xyz: list[float]) -> list[dict]:
        self.load(state, scene)
        start = self.data.xpos[self.model.body(TCP_FRAMES[side]).id].copy()
        end = np.asarray(xyz, dtype=float)
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.025)))
        # Plan the entire path before dispatch, so an unreachable endpoint does
        # not leave a half-executed tool. Joint and lift limits are enforced above.
        return [self.solve(side, start + (end - start) * i / count) for i in range(1, count + 1)]
