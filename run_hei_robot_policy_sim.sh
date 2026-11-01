#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  exec "$ROOT/run_hei_robot_policy_sim_on_mac.sh" "$@"
fi
cd "$ROOT"
exec env -u LD_LIBRARY_PATH uv run --locked --project "$ROOT" python -u \
  "$ROOT/software/lerobot-hei-rebot-lift/examples/hei_rebot_lift/VR_mujoco_ik/mujoco_ik/hei_robot_mujoco_zmq_server.py" \
  --control-source policy --publish-fps 10 "$@"
