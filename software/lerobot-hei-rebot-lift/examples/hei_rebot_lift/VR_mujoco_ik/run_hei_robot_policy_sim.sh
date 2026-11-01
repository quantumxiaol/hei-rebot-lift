#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${HEI_REBOT_VR_CONDA_ENV:-hei-rebot-vr}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Prefer the standalone CPU-only uv project; retain the existing Conda entry
# for installations that explicitly select it or do not have uv installed.
if [[ "${HEI_SIM_RUNTIME:-uv}" == "uv" ]] && command -v uv >/dev/null 2>&1 \
  && [[ -f "$PROJECT_ROOT/run_hei_robot_policy_sim.sh" ]]; then
  exec "$PROJECT_ROOT/run_hei_robot_policy_sim.sh" "$@"
fi

if [[ -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != "$ENV_NAME" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$("$CONDA_EXE" info --base)"
  else
    CONDA_BASE="$(conda info --base)"
  fi
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

cd "$SCRIPT_DIR/mujoco_ik"
env -u LD_LIBRARY_PATH python -u hei_robot_mujoco_zmq_server.py --control-source policy "$@"
