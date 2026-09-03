"""Small, dependency-light wrapper around the XRoboToolkit PC SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BUTTON_NAMES = (
    "A",
    "B",
    "X",
    "Y",
    "left_menu",
    "right_menu",
    "left_axis_click",
    "right_axis_click",
)


@dataclass(frozen=True)
class XRState:
    """One synchronized controller sample from XRoboToolkit."""

    timestamp_ns: int
    controller_pose_xyzw: np.ndarray
    grip: np.ndarray
    trigger: np.ndarray
    axis: np.ndarray
    buttons: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_pose_xyzw",
            np.asarray(self.controller_pose_xyzw, dtype=np.float64).reshape(2, 7),
        )
        object.__setattr__(self, "grip", np.asarray(self.grip, dtype=np.float32).reshape(2))
        object.__setattr__(
            self,
            "trigger",
            np.asarray(self.trigger, dtype=np.float32).reshape(2),
        )
        object.__setattr__(self, "axis", np.asarray(self.axis, dtype=np.float32).reshape(2, 2))
        object.__setattr__(self, "buttons", np.asarray(self.buttons, dtype=np.bool_).reshape(8))

    def button(self, name: str) -> bool:
        try:
            index = BUTTON_NAMES.index(str(name))
        except ValueError as exc:
            raise KeyError(f"Unknown XR button: {name}") from exc
        return bool(self.buttons[index])


class XRClient:
    """Read PICO controller state without importing the sample teleop stack."""

    def __init__(self, sdk: Any | None = None) -> None:
        if sdk is None:
            try:
                import xrobotoolkit_sdk as sdk_module
            except ImportError as exc:
                raise ImportError(
                    "xrobotoolkit_sdk is not installed in this Python environment. "
                    "Build the XRoboToolkit PC-Service pybind package for the active "
                    "Python version before running VR teleoperation."
                ) from exc
            sdk = sdk_module
        self._sdk = sdk
        self._closed = False
        self._sdk.init()

    def read(self) -> XRState:
        if self._closed:
            raise RuntimeError("XRClient is closed.")
        controller_pose = np.stack(
            [
                np.asarray(self._sdk.get_left_controller_pose(), dtype=np.float64),
                np.asarray(self._sdk.get_right_controller_pose(), dtype=np.float64),
            ],
            axis=0,
        )
        grip = np.asarray(
            [self._sdk.get_left_grip(), self._sdk.get_right_grip()],
            dtype=np.float32,
        )
        trigger = np.asarray(
            [self._sdk.get_left_trigger(), self._sdk.get_right_trigger()],
            dtype=np.float32,
        )
        axis = np.stack(
            [self._sdk.get_left_axis(), self._sdk.get_right_axis()],
            axis=0,
        ).astype(np.float32, copy=False)
        buttons = np.asarray(
            [
                self._sdk.get_A_button(),
                self._sdk.get_B_button(),
                self._sdk.get_X_button(),
                self._sdk.get_Y_button(),
                self._sdk.get_left_menu_button(),
                self._sdk.get_right_menu_button(),
                self._sdk.get_left_axis_click(),
                self._sdk.get_right_axis_click(),
            ],
            dtype=np.bool_,
        )
        return XRState(
            timestamp_ns=int(self._sdk.get_time_stamp_ns()),
            controller_pose_xyzw=controller_pose,
            grip=grip,
            trigger=trigger,
            axis=axis,
            buttons=buttons,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._sdk.close()
        self._closed = True

    def __enter__(self) -> "XRClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
