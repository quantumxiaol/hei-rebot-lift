"""Locate the existing standalone simulator without importing the LeRobot package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "software/lerobot-hei-rebot-lift/examples/hei_rebot_lift/VR_mujoco_ik/mujoco_ik"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
