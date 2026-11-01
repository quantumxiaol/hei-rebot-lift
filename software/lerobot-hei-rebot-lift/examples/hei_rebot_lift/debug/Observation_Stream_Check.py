#!/usr/bin/env python
"""Measure HEI observation delivery without opening serial ports or sending actions.

Keep the robot host running, but stop all rollout/observation clients first:
the host's PUSH socket distributes frames between clients rather than broadcasting.
Run this separately on the Jetson (127.0.0.1) and the inference server (Jetson IP).
This measures idle-host receive timing, not image capture age or inference latency.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

try:
    import zmq
except ImportError:
    zmq = None


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("Must be finite and greater than zero.")
    return seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Robot host IP, not the GPU server IP.")
    parser.add_argument("--port", type=int, default=6556, help="Observation port (default: 6556).")
    parser.add_argument("--duration-sec", type=positive_seconds, default=15.0)
    parser.add_argument("--connect-timeout-sec", type=positive_seconds, default=5.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def describe_message(parts: list[bytes]) -> str:
    """Inspect the JSON header only; never decode or save camera images."""
    try:
        header = json.loads(parts[0])
        if not isinstance(header, dict):
            raise ValueError("JSON header is not an object")
        state_keys = sorted(key for key in header if key.endswith((".pos", ".vel")))
        if "_cams" in header:
            cameras = header["_cams"]
            if not isinstance(cameras, list) or not all(isinstance(name, str) for name in cameras):
                raise ValueError("_cams is not a list of names")
            if len(parts) != len(cameras) + 1:
                raise ValueError("camera names and multipart image count differ")
            empty = [name for name, data in zip(cameras, parts[1:]) if not data]
            protocol = "multipart"
        else:
            cameras = [name for name in ("front", "left_wrist", "right_wrist") if name in header]
            empty = [name for name in cameras if not header[name]]
            protocol = "legacy JSON"
        return (
            f"First header: protocol={protocol}, state_fields={len(state_keys)}, "
            f"cameras={cameras}, empty_images={empty}\n"
            f"State keys: {', '.join(state_keys)}"
        )
    except (IndexError, UnicodeDecodeError, ValueError, TypeError) as exc:
        return f"First header could not be parsed: {exc}"


def run(args: argparse.Namespace) -> int:
    if zmq is None:
        raise RuntimeError("pyzmq is required; use the existing robot or VLA Python environment.")

    endpoint = f"tcp://{args.host}:{args.port}"
    print(f"Receive only: {endpoint}; no commands or serial access.", flush=True)
    print("Keep ONLY this observation client connected; leave the robot host running.", flush=True)
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    count = total_bytes = 0
    first_rx: float | None = None
    last_rx: float | None = None
    intervals: list[float] = []
    started = time.monotonic()
    deadline = started + args.connect_timeout_sec
    next_report = started + 1.0
    try:
        receiver.setsockopt(zmq.LINGER, 0)
        receiver.setsockopt(zmq.RCVHWM, 2)
        receiver.connect(endpoint)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            events = receiver.poll(timeout=max(1, math.ceil(min(0.2, remaining) * 1000)))
            now = time.monotonic()
            if now >= deadline:
                break
            if events & zmq.POLLIN:
                try:
                    parts = receiver.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                now = time.monotonic()
                if first_rx is None:
                    first_rx = now
                    deadline = now + args.duration_sec
                    next_report = now + 1.0
                    print(f"First observation after {now - started:.3f} s", flush=True)
                    print(describe_message(parts), flush=True)
                elif last_rx is not None:
                    intervals.append(now - last_rx)
                last_rx = now
                count += 1
                total_bytes += sum(len(part) for part in parts)
            if now >= next_report:
                if first_rx is not None and last_rx is not None:
                    elapsed = now - first_rx
                    print(
                        f"{elapsed:5.1f}s: {count} observations, {count / elapsed:.2f} Hz, "
                        f"last receive {(now - last_rx) * 1000:.1f} ms ago",
                        flush=True,
                    )
                else:
                    print("Waiting for first observation...", flush=True)
                next_report = now + 1.0
        finished = time.monotonic()
    finally:
        receiver.close(linger=0)
        context.term()

    if first_rx is None or last_rx is None:
        print(f"NO OBSERVATIONS within {args.connect_timeout_sec:g} s.", flush=True)
        return 1

    elapsed = finished - first_rx
    end_silence = finished - last_rx
    longest_gap = max([end_silence, *intervals])
    print(f"Summary: {count} observations / {elapsed:.2f} s = {count / elapsed:.2f} Hz")
    print(f"Payload rate: {total_bytes * 8 / elapsed / 1e6:.2f} Mbit/s (excludes transport overhead)")
    if intervals:
        p95 = sorted(intervals)[math.ceil(len(intervals) * 0.95) - 1]
        print(
            f"Receive intervals: median={statistics.median(intervals) * 1000:.1f} ms, "
            f"p95={p95 * 1000:.1f} ms"
        )
    print(f"Longest receive gap (including silence at end): {longest_gap * 1000:.1f} ms")
    print(f"Last observation received {end_silence * 1000:.1f} ms before measurement ended")
    print("Arrival timing only: host messages have no capture timestamp or sequence number.")
    return 0


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit(result)


if __name__ == "__main__":
    main()
