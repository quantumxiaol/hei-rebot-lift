"""Exercise the real MuJoCo model and action adapter without API calls or a viewer."""

import unittest
from types import SimpleNamespace

import numpy as np

from sim_agent import paths  # noqa: F401
from sim_agent.client import Frame
from sim_agent.controller import Controller
from sim_agent.kinematics import Kinematics

# isort: split
from hei_robot_mujoco_dataset_adapter import HEIMujocoDatasetAdapter
from hei_robot_mujoco_scene import SceneSettings
from hei_robot_vr_mujoco_sim import DEFAULT_URDF, HEIRobotVRSimulator


class PolicySimulation(HEIRobotVRSimulator):
    def _build_arm_solvers(self) -> dict:
        return {}

    def _vr_listener(self) -> None:
        pass


def make_simulation() -> PolicySimulation:
    return PolicySimulation(
        SimpleNamespace(
            model=DEFAULT_URDF,
            plain_scene=False,
            scene_settings=SceneSettings(),
            initial_height_mm=0.0,
            lift_speed_m_s=0.20,
            verbose=False,
        )
    )


def reach(adapter: HEIMujocoDatasetAdapter, changes: dict) -> None:
    action = adapter.read_state()
    action.update(changes)
    for _ in range(1200):
        state = adapter.apply_action(action, 1 / 60)
        if all(
            abs(state[key] - value) < (0.5 if key == "height.pos" else 0.001)
            for key, value in changes.items()
        ):
            return
    raise AssertionError("Action did not converge")


def move(adapter: HEIMujocoDatasetAdapter, ik: Kinematics, side: str, xyz: list[float]) -> None:
    snapshot = adapter.capture_snapshot()
    for waypoint in ik.path(snapshot.state, snapshot.simulation, side, xyz):
        reach(adapter, waypoint)


class PickupTest(unittest.TestCase):
    def test_closing_above_cube_does_not_attach_or_lift_it(self) -> None:
        sim = make_simulation()
        adapter = HEIMujocoDatasetAdapter(sim, create_renderer=False)
        try:
            ik = Kinematics()
            initial = adapter.capture_snapshot().simulation["objects"]["cube_red"]["position"]
            for clearance in (0.10, 0.03):
                with self.subTest(clearance=clearance):
                    reach(adapter, {"right_gripper.pos": -4.5})
                    move(adapter, ik, "right", [initial[0], initial[1], initial[2] + clearance])
                    reach(adapter, {"right_gripper.pos": 0.0})
                    self.assertIsNone(
                        adapter.capture_snapshot().simulation["objects"]["cube_red"]["held_by"]
                    )
                    move(adapter, ik, "right", [initial[0], initial[1], initial[2] + 0.20])
                    np.testing.assert_allclose(
                        adapter.capture_snapshot().simulation["objects"]["cube_red"]["position"],
                        initial,
                    )
        finally:
            adapter.close()
            sim.close()

    def test_pickup_with_real_ik_and_gripper_edges(self) -> None:
        for object_name, side in (("cube_blue", "left"), ("cube_red", "right"), ("cube_green", "left")):
            with self.subTest(object=object_name, arm=side):
                self.check_pickup(object_name, side)

    def check_pickup(self, object_name: str, side: str) -> None:
        sim = make_simulation()
        adapter = HEIMujocoDatasetAdapter(sim, create_renderer=False)
        try:
            ik = Kinematics()
            initial = adapter.capture_snapshot().simulation
            xyz = initial["objects"][object_name]["position"]
            reach(adapter, {f"{side}_gripper.pos": -4.5})
            move(adapter, ik, side, [xyz[0], xyz[1], xyz[2] + 0.12])
            move(adapter, ik, side, xyz)
            # Reaching a cube with an open gripper must not attach it.
            self.assertIsNone(adapter.capture_snapshot().simulation["objects"][object_name]["held_by"])
            reach(adapter, {f"{side}_gripper.pos": 0.0})
            captured = adapter.capture_snapshot().simulation
            self.assertEqual(captured["objects"][object_name]["held_by"], side)
            tcp = captured["tcp"][side]
            self.assertLessEqual(
                np.linalg.norm(np.asarray(captured["objects"][object_name]["position"]) - tcp), 0.015
            )
            move(adapter, ik, side, [tcp[0], tcp[1], tcp[2] + 0.15])
            lifted = adapter.capture_snapshot().simulation["objects"][object_name]
            self.assertGreaterEqual(lifted["position"][2] - lifted["initial_position"][2], 0.10)
            reach(adapter, {f"{side}_gripper.pos": -4.5})
            self.assertIsNone(adapter.capture_snapshot().simulation["objects"][object_name]["held_by"])
        finally:
            adapter.close()
            sim.close()

    def test_unreachable_target_does_not_modify_simulator(self) -> None:
        sim = make_simulation()
        adapter = HEIMujocoDatasetAdapter(sim, create_renderer=False)
        try:
            snapshot = adapter.capture_snapshot()
            ik = Kinematics()
            with self.assertRaises(ValueError):
                ik.path(snapshot.state, snapshot.simulation, "left", [5.0, 0.0, 0.8])
            np.testing.assert_array_equal(sim.data.qpos, snapshot.qpos)
        finally:
            adapter.close()
            sim.close()


class TaskGoalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.objects = {
            name: {"position": [0.7, y, 0.82], "initial_position": [0.7, y, 0.82], "held_by": None}
            for name, y in (("cube_blue", 0.28), ("cube_red", -0.28))
        }
        self.tcp = {"right": [0.7, -0.28, 0.98], "left": [0.7, 0.28, 1.02]}
        scene = {"session_id": "test", "reset_count": 0, "objects": self.objects, "tcp": self.tcp}
        frame = Frame({"simulation": scene}, (), 0.0)
        self.commands = []
        self.controller = Controller(
            SimpleNamespace(frame=lambda: frame, hold=lambda: None, set_target=self.commands.append)
        )

    def test_close_from_above_is_rejected_before_sending_a_command(self) -> None:
        self.controller.execute("select_target", {"object_name": "cube_red", "minimum_lift_m": 0.1})
        self.tcp["right"] = [0.7, -0.28, 0.92]
        with self.assertRaisesRegex(ValueError, "clearance_m=0"):
            self.controller.execute("set_gripper", {"arm": "right", "opening": "closed"})
        self.assertEqual(self.commands, [])

    def test_done_rejects_a_distant_attached_object_from_an_old_simulator(self) -> None:
        self.controller.execute("select_target", {"object_name": "cube_red", "minimum_lift_m": 0.1})
        self.objects["cube_red"].update(held_by="right", position=[0.7, -0.28, 0.98], grasp_radius_m=0.11)
        self.tcp["right"] = [0.7, -0.28, 1.08]
        self.assertFalse(self.controller.execute("done", {})["success"])

    def test_goal_is_required_and_cannot_be_replaced_or_weakened(self) -> None:
        with self.assertRaises(ValueError):
            self.controller.execute("set_gripper", {"arm": "right", "opening": "open"})
        with self.assertRaises(ValueError):
            self.controller.execute("done", {})
        self.controller.execute("select_target", {"object_name": "cube_red", "minimum_lift_m": 0.15})
        for arguments in (
            {"object_name": "cube_blue", "minimum_lift_m": 0.15},
            {"object_name": "cube_red", "minimum_lift_m": 0.10},
        ):
            with self.assertRaises(ValueError):
                self.controller.execute("select_target", arguments)
        with self.assertRaises(ValueError):
            self.controller.execute("move_to_object", {"arm": "right", "object_name": "cube_blue"})

    def test_done_checks_selected_object_and_height(self) -> None:
        self.controller.execute("select_target", {"object_name": "cube_red", "minimum_lift_m": 0.15})
        self.objects["cube_blue"].update(held_by="left", position=[0.7, 0.28, 1.02])
        self.assertFalse(self.controller.execute("done", {})["success"])
        self.objects["cube_red"].update(held_by="right", position=[0.7, -0.28, 0.96])
        self.assertFalse(self.controller.execute("done", {})["success"])
        self.objects["cube_red"]["position"][2] = 0.98
        self.assertTrue(self.controller.execute("done", {})["success"])
        self.objects["cube_red"]["held_by"] = None
        self.assertFalse(self.controller.execute("done", {})["success"])


if __name__ == "__main__":
    unittest.main()
