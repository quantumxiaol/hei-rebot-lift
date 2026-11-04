#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Use ./run_hei_robot_policy_sim.sh on Linux." >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
cd "$ROOT"
# mjpython keeps the Cocoa viewer on the main thread. CGL renders the three
# cameras offscreen; neither Xvfb nor Linux GLX environment settings are needed.
exec env -u DISPLAY -u PYOPENGL_PLATFORM -u LIBGL_ALWAYS_SOFTWARE -u LD_LIBRARY_PATH \
  MUJOCO_GL=cgl uv run --locked --project "$ROOT" python -m sim_agent.mac_viewer -u \
  "$ROOT/software/lerobot-hei-rebot-lift/examples/hei_rebot_lift/VR_mujoco_ik/mujoco_ik/hei_robot_mujoco_zmq_server.py" \
  --control-source policy --bind-ip 127.0.0.1 --publish-fps 10 --camera-preview "$@"
