"""Display the published RGB camera frames in the native MuJoCo viewer."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import mujoco
import numpy as np

if TYPE_CHECKING:
    from hei_robot_mujoco_observation_publisher import CameraPreviewFrame
    from mujoco.viewer import Handle


class CameraPreview:
    """A right-hand camera strip, redrawn only for new frames or window changes."""

    def __init__(self) -> None:
        self._last_layout: tuple | None = None
        self._overlay_initialized = False

    def update(self, viewer: Handle, frame: CameraPreviewFrame | None) -> None:
        with viewer.lock():
            if not viewer.is_running():
                return
            viewport = viewer.viewport
            if viewport is None:
                return
            left, bottom, width, height = viewport.left, viewport.bottom, viewport.width, viewport.height
        stale = frame is None or time.time() - frame.timestamp_s > 2.0
        layout = (left, bottom, width, height, frame.sequence if frame else None, stale)
        if not self._overlay_initialized:
            # MuJoCo 3.5 draws images without resetting the 3D depth test.
            # A text overlay initializes 2D drawing even when both UI panels
            # are hidden; otherwise camera pixels can fail the depth test.
            viewer.set_texts(
                [
                    (
                        mujoco.mjtFont.mjFONT_NORMAL,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "R: reset    Tab: controls",
                        "",
                    )
                ]
            )
            self._overlay_initialized = True
        if layout == self._last_layout:
            return
        self._last_layout = layout
        if width < 480 or height < 320:
            viewer.clear_images()
            return

        margin = max(6, int(height * 0.012))
        panel_width = int(min(width * 0.30, height * 0.42, 480))
        panel_height = height - 2 * margin
        panel = np.full((panel_height, panel_width, 3), (20, 25, 33), dtype=np.uint8)
        scale = max(0.4, min(1.0, panel_width / 420))
        pad = max(6, int(12 * scale))
        header = max(28, int(48 * scale))
        label_height = max(22, int(34 * scale))
        row_height = (panel_height - header - pad) // 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            panel, "CAMERAS", (pad, int(header * 0.65)), font, 0.72 * scale, (232, 237, 244), 1, cv2.LINE_AA
        )
        status = "WAITING" if frame is None else "STALE" if stale else "LIVE"
        status_width = cv2.getTextSize(status, font, 0.50 * scale, 1)[0][0]
        cv2.putText(
            panel,
            status,
            (panel_width - pad - status_width, int(header * 0.65)),
            font,
            0.50 * scale,
            (240, 180, 85) if stale else (95, 220, 165),
            1,
            cv2.LINE_AA,
        )

        for index, label in enumerate(("01  FRONT", "02  LEFT WRIST", "03  RIGHT WRIST")):
            top = header + index * row_height
            cv2.putText(
                panel,
                label,
                (pad, top + int(label_height * 0.72)),
                font,
                0.60 * scale,
                (190, 205, 222),
                1,
                cv2.LINE_AA,
            )
            image_top = top + label_height
            box_width = panel_width - 2 * pad
            box_height = row_height - label_height - pad
            panel[image_top : image_top + box_height, pad : pad + box_width] = (8, 12, 17)
            if frame is None:
                continue
            rgb = frame.images[index]
            ratio = min(box_width / rgb.shape[1], box_height / rgb.shape[0])
            image_width = max(1, int(rgb.shape[1] * ratio))
            image_height = max(1, int(rgb.shape[0] * ratio))
            resized = cv2.resize(rgb, (image_width, image_height), interpolation=cv2.INTER_AREA)
            x = pad + (box_width - image_width) // 2
            y = image_top + (box_height - image_height) // 2
            panel[y : y + image_height, x : x + image_width] = resized

        # Handle.set_images expects top-down RGB and flips it for OpenGL itself.
        rect = mujoco.MjrRect(left + width - panel_width - margin, bottom + margin, panel_width, panel_height)
        viewer.set_images([(rect, panel)])
