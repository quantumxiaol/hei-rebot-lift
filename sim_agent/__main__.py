"""Run a DeepSeek vision/tool loop, or an explicitly selected offline demo plan."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .client import SimulationClient
from .controller import DEMO_PLAN, Controller
from .paths import ROOT


def tool(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


ARM = {"type": "string", "enum": ["left", "right"]}
DEFAULT_TASK = "Pick up the blue cube and hold it at least 0.10 m above its starting height."
TOOLS = [
    tool("observe", "Read current simulator ground truth: objects, TCPs, and holding status.", {}),
    tool(
        "select_target",
        "Before moving, interpret the user's task and select its scene object and minimum lift. "
        "The goal is then locked for this run. If no height was requested, use 0.10 m.",
        {
            "object_name": {"type": "string"},
            "minimum_lift_m": {"type": "number", "minimum": 0.05, "maximum": 0.25},
        },
    ),
    tool(
        "set_gripper",
        "Open or close one gripper. A grasp needs an open-to-closed transition near the object.",
        {"arm": ARM, "opening": {"type": "string", "enum": ["open", "closed"]}},
    ),
    tool(
        "move_to_object",
        "Move TCP to a named object's centre plus vertical clearance, using local IK.",
        {
            "arm": ARM,
            "object_name": {"type": "string"},
            "clearance_m": {"type": "number", "minimum": 0, "maximum": 0.25},
        },
    ),
    tool(
        "lift",
        "Raise an already held object vertically; keep the gripper closed.",
        {"arm": ARM, "height_m": {"type": "number", "minimum": 0.05, "maximum": 0.25}},
    ),
    tool("done", "Verify that the selected object is held and raised by at least the selected minimum.", {}),
    tool(
        "cannot_complete",
        "Stop and report an unsupported, ambiguous, or failed task without claiming success.",
        {"reason": {"type": "string"}},
    ),
]

SYSTEM = """You control ONLY a local HEI MuJoCo simulation using tools.
The user supplies the task separately. Interpret it using the current scene and images.
Supported tasks: pick up ONE named/visually described scene object and hold it raised 0.05–0.25 m.
Do not silently simplify placement, stacking, multi-object, or other unsupported tasks into a pickup.
If the request is unsupported, ambiguous, or impossible, use cannot_complete with an explanation.
Before any movement, call select_target with the scene object's exact name and the requested minimum lift.
If the task gives no lift height, use a 0.10 m minimum. This goal cannot be changed during the run.
Choose the arm and tool sequence yourself from observations. Positive world Y is the robot's left;
negative Y is its right. Objects near Y=0 can be approached with either arm.
The gripper must be open during approach; approaching from above gives table clearance.
Grasping requires an open-to-closed transition near the object's centre, then feedback confirming held_by.
Plan enough lift to meet the selected minimum, allowing for small IK position errors.
Each tool is blocking and returns measured simulator feedback. Choose one tool per turn.
If grasp fails, reopen before retrying. Never claim success without done.success=true.
Keep the selected object held after lifting. Images are front, left_wrist, right_wrist.
The JSON is explicit simulator ground truth, not a real-world vision estimate.
The existing simulator uses assisted kinematic grasping, not contact-force physics.
Do not output joint angles, code, or requests to access a real robot.
"""


def observation_message(controller: Controller, images: bool) -> dict:
    frame = controller.frame()
    content = [
        {
            "type": "text",
            "text": "Current observation: "
            + json.dumps(
                {
                    "scene": frame.scene,
                    "state": frame.state,
                    "goal": controller.goal_info(),
                },
                ensure_ascii=False,
            ),
        }
    ]
    if images:
        for name, jpeg in zip(("front", "left_wrist", "right_wrist"), frame.images, strict=True):
            content.append({"type": "text", "text": f"Camera: {name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
                        "detail": "low",
                    },
                }
            )
    return {"role": "user", "content": content}


def strip_old_images(messages: list[dict]) -> None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [item for item in content if item.get("type") != "image_url"]


def run_deepseek(controller: Controller, steps: int, log_path: Path, text_only: bool, task: str) -> bool:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(f"Fill DEEPSEEK_API_KEY in {ROOT / '.env'}, or explicitly use --demo.")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
    images = not text_only and os.environ.get("HEI_AGENT_SEND_IMAGES", "true").lower() in ("1", "true", "yes")
    timeout = float(os.environ.get("DEEPSEEK_TIMEOUT_SEC", "60"))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    print(f"[Agent] DeepSeek model={model}, images={images}, max_steps={steps}", flush=True)
    print(f"[Agent] Task: {task}", flush=True)
    with httpx.Client(timeout=timeout) as api, log_path.open("a", encoding="utf-8") as trace:
        for index in range(steps):
            strip_old_images(messages)
            messages.append(observation_message(controller, images))
            print(f"[Agent] Asking DeepSeek, turn {index + 1}/{steps}...", flush=True)
            response = api.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "thinking": {"type": "disabled"},
                    "max_tokens": 1024,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
            if response.is_error:
                detail = response.text[:1000].replace(key, "[REDACTED]")
                raise RuntimeError(f"DeepSeek HTTP {response.status_code}: {detail}")
            body = response.json()
            message = body["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    **({"tool_calls": calls} if calls else {}),
                }
            )
            if message.get("content"):
                print(f"[DeepSeek] {message['content']}", flush=True)
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Use tools to continue, verify with done, or use cannot_complete.",
                    }
                )
                continue
            done = False
            aborted = False
            for call in calls:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise TypeError("Tool arguments must be a JSON object")
                    print(f"[Tool] {name} {json.dumps(arguments, ensure_ascii=False)}", flush=True)
                    result = controller.execute(name, arguments)
                except (ValueError, TypeError, KeyError, TimeoutError) as exc:
                    controller.client.hold()
                    result = {"error": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                trace.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "task": task,
                            "turn": index + 1,
                            "tool": name,
                            "arguments": call["function"]["arguments"],
                            "result": result,
                            "usage": body.get("usage"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                trace.flush()
                done = done or (name == "done" and result.get("success") is True)
                if name == "cannot_complete" and "reason" in result:
                    print(f"[Agent] Cannot complete: {result['reason']}", flush=True)
                    aborted = True
                    break
            if aborted:
                return False
            if done and controller.success():
                return True
    return False


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", help="Use the fixed local test plan; do not call DeepSeek."
    )
    parser.add_argument(
        "--text-only", action="store_true", help="Send structured simulator state without images."
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--task", help=f"Natural-language pickup task. Default: {DEFAULT_TASK}")
    args = parser.parse_args()
    if args.task is not None and not args.task.strip():
        parser.error("--task must not be empty")
    if args.demo and args.task is not None:
        parser.error("--demo uses the fixed blue-cube plan; omit --demo to run a custom --task with DeepSeek")
    task = args.task.strip() if args.task is not None else DEFAULT_TASK
    if not 1 <= args.max_steps <= 30:
        parser.error("--max-steps must be between 1 and 30")
    if not args.demo and not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        parser.error("Fill DEEPSEEK_API_KEY in the root .env, or use --demo for a fixed-plan test.")
    log_dir = ROOT / "outputs/sim-agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    try:
        with SimulationClient() as client:
            controller = Controller(client)
            if args.demo:
                print("[Agent] DEMO: fixed tool plan, no language model/API call", flush=True)
                for name, arguments in DEMO_PLAN:
                    print(f"[Tool] {name} {arguments}", flush=True)
                    controller.execute(name, arguments)
                success = controller.success()
            else:
                success = run_deepseek(controller, args.max_steps, log_path, args.text_only, task)
                print(f"[Agent] Tool trace: {log_path}", flush=True)
            print(
                f"[Agent] {'SUCCESS' if success else 'NOT COMPLETE'}: "
                f"goal={json.dumps(controller.goal_info(), ensure_ascii=False)}",
                flush=True,
            )
            if not success:
                raise SystemExit(1)
    except KeyboardInterrupt:
        print("[Agent] Interrupted; holding the simulator's current pose.")
        raise SystemExit(130) from None
    except Exception as exc:
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        message = str(exc).replace(key, "[REDACTED]") if key else str(exc)
        raise SystemExit(f"[Agent] {message}") from exc


if __name__ == "__main__":
    main()
