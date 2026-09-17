"""Sanity checks for calibration validation.

This module provides functions to validate calibration results
against expected physical constraints and assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from hsi_rgbd_calib.common.logging import get_logger
from hsi_rgbd_calib.common.transforms import decompose_transform

logger = get_logger(__name__)


@dataclass
class SlowMotionCheckResult:
    """Result of slow-motion assumption validation.
    
    The slow-motion assumption requires that during HSI acquisition,
    the rig moves slowly enough that adjacent HSI lines can be considered
    as sampling the same static scene.
    
    Attributes:
        passed: Whether the check passed.
        max_motion_per_line_mm: Maximum estimated motion per HSI line.
        threshold_mm: Motion threshold used.
        fps: HSI frame rate.
        estimated_velocity_mps: Estimated rig velocity.
        notes: Additional notes.
    """
    
    passed: bool
    max_motion_per_line_mm: float
    threshold_mm: float
    fps: float
    estimated_velocity_mps: float
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "max_motion_per_line_mm": self.max_motion_per_line_mm,
            "threshold_mm": self.threshold_mm,
            "fps": self.fps,
            "estimated_velocity_mps": self.estimated_velocity_mps,
            "notes": self.notes,
        }


def check_slow_motion_assumption(
    hsi_fps: float,
    estimated_velocity_mps: float,
    threshold_mm: float = 1.0,
) -> SlowMotionCheckResult:
    """Check if the slow-motion assumption is satisfied.
    
    For pushbroom HSI cameras, we assume quasi-static acquisition where
    motion between adjacent scan lines is negligible. This function
    validates that assumption.
    
    Args:
        hsi_fps: HSI line rate (lines per second).
        estimated_velocity_mps: Estimated rig velocity (m/s).
        threshold_mm: Maximum acceptable motion per line (mm).
        
    Returns:
        SlowMotionCheckResult with validation details.
    """
    if hsi_fps <= 0:
        return SlowMotionCheckResult(
            passed=False,
            max_motion_per_line_mm=np.inf,
            threshold_mm=threshold_mm,
            fps=hsi_fps,
            estimated_velocity_mps=estimated_velocity_mps,
            notes="Invalid FPS (must be positive)",
        )
    
    # Motion per line = velocity / fps
    motion_per_line_m = estimated_velocity_mps / hsi_fps
    motion_per_line_mm = motion_per_line_m * 1000
    
    passed = motion_per_line_mm <= threshold_mm
    
    notes = ""
    if passed:
        notes = f"Motion {motion_per_line_mm:.3f}mm/line is within threshold"
    else:
        notes = (
            f"Motion {motion_per_line_mm:.3f}mm/line exceeds threshold. "
            f"Consider reducing rig velocity or increasing FPS."
        )
    
    return SlowMotionCheckResult(
        passed=passed,
        max_motion_per_line_mm=motion_per_line_mm,
        threshold_mm=threshold_mm,
        fps=hsi_fps,
        estimated_velocity_mps=estimated_velocity_mps,
        notes=notes,
    )


@dataclass
class ExtrinsicSanityResult:
    """Result of extrinsic sanity check.
    
    Attributes:
        passed: Whether the check passed.
        translation_magnitude_m: Translation magnitude.
        rotation_angle_deg: Rotation angle from identity.
        translation_in_bounds: Whether translation is within expected bounds.
        rotation_in_bounds: Whether rotation is within expected bounds.
        notes: Additional notes.
    """
    
    passed: bool
    translation_magnitude_m: float
    rotation_angle_deg: float
    translation_in_bounds: bool
    rotation_in_bounds: bool
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "translation_magnitude_m": self.translation_magnitude_m,
            "rotation_angle_deg": self.rotation_angle_deg,
            "translation_in_bounds": self.translation_in_bounds,
            "rotation_in_bounds": self.rotation_in_bounds,
            "notes": self.notes,
        }


def check_extrinsic_sanity(
    T: NDArray[np.float64],
    max_translation_m: float = 0.5,
    max_rotation_deg: float = 45.0,
) -> ExtrinsicSanityResult:
    """Check if extrinsic calibration is within expected bounds.
    
    For a typical HSI + RGB camera rig, we expect:
    - Translation to be within 0.5m (cameras are mounted close together)
    - Rotation to be within 45 degrees (cameras point roughly same direction)
    
    Args:
        T: 4x4 extrinsic transformation matrix.
        max_translation_m: Maximum expected translation (m).
        max_rotation_deg: Maximum expected rotation (degrees).
        
    Returns:
        ExtrinsicSanityResult with validation details.
    """
    R, t = decompose_transform(T)
    
    # Check translation
    translation_magnitude = float(np.linalg.norm(t))
    translation_in_bounds = translation_magnitude <= max_translation_m
    
    # Check rotation (angle from identity)
    angle_rad = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    rotation_angle_deg = float(np.degrees(angle_rad))
    rotation_in_bounds = rotation_angle_deg <= max_rotation_deg
    
    passed = translation_in_bounds and rotation_in_bounds
    
    notes_parts = []
    if not translation_in_bounds:
        notes_parts.append(
            f"Translation {translation_magnitude:.3f}m exceeds {max_translation_m}m"
        )
    if not rotation_in_bounds:
        notes_parts.append(
            f"Rotation {rotation_angle_deg:.1f}° exceeds {max_rotation_deg}°"
        )
    
    if passed:
        notes = "Extrinsic calibration is within expected bounds"
    else:
        notes = "; ".join(notes_parts)
    
    return ExtrinsicSanityResult(
        passed=passed,
        translation_magnitude_m=translation_magnitude,
        rotation_angle_deg=rotation_angle_deg,
        translation_in_bounds=translation_in_bounds,
        rotation_in_bounds=rotation_in_bounds,
        notes=notes,
    )


def check_intrinsic_sanity(
    focal_length: float,
    principal_point: float,
    image_size: int,
    min_focal_ratio: float = 0.5,
    max_focal_ratio: float = 3.0,
) -> Dict[str, Any]:
    """Check if intrinsic parameters are within expected bounds.
    
    Args:
        focal_length: Focal length in pixels.
        principal_point: Principal point in pixels.
        image_size: Image dimension (width for slit).
        min_focal_ratio: Minimum focal length / image size ratio.
        max_focal_ratio: Maximum focal length / image size ratio.
        
    Returns:
        Dictionary with validation results.
    """
    focal_ratio = focal_length / image_size
    pp_offset = abs(principal_point - image_size / 2)
    pp_offset_ratio = pp_offset / (image_size / 2)
    
    focal_in_bounds = min_focal_ratio <= focal_ratio <= max_focal_ratio
    pp_in_bounds = pp_offset_ratio <= 0.1  # Principal point within 10% of center
    
    passed = focal_in_bounds and pp_in_bounds
    
    notes = []
    if not focal_in_bounds:
        notes.append(
            f"Focal ratio {focal_ratio:.2f} outside [{min_focal_ratio}, {max_focal_ratio}]"
        )
    if not pp_in_bounds:
        notes.append(
            f"Principal point offset {pp_offset:.1f}px ({pp_offset_ratio*100:.1f}%) is large"
        )
    
    return {
        "passed": passed,
        "focal_length": focal_length,
        "principal_point": principal_point,
        "focal_ratio": focal_ratio,
        "pp_offset_pixels": pp_offset,
        "notes": "; ".join(notes) if notes else "Intrinsics are within expected bounds",
    }
