"""Common utilities for HSI-RGBD calibration."""

from hsi_rgbd_calib.common.transforms import (
    compose,
    invert,
    apply_transform,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    make_transform,
    decompose_transform,
)
from hsi_rgbd_calib.common.frames import Frame, FRAME_CONVENTIONS
from hsi_rgbd_calib.common.logging import setup_logging, get_logger

__all__ = [
    # Transforms
    "compose",
    "invert",
    "apply_transform",
    "rotation_matrix_to_euler",
    "euler_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "make_transform",
    "decompose_transform",
    # Frames
    "Frame",
    "FRAME_CONVENTIONS",
    # Logging
    "setup_logging",
    "get_logger",
]
