"""Repeatability evaluation for calibration.

This module provides functions to assess calibration repeatability
by comparing multiple calibration runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.common.logging import get_logger
from hsi_rgbd_calib.common.transforms import decompose_transform

logger = get_logger(__name__)


@dataclass
class RepeatabilityMetrics:
    """Repeatability metrics from multiple calibration runs.
    
    Attributes:
        translation_mean_m: Mean translation (m).
        translation_std_m: Standard deviation of translation (m).
        rotation_mean_deg: Mean rotation (degrees).
        rotation_std_deg: Standard deviation of rotation (degrees).
        num_calibrations: Number of calibrations compared.
    """
    
    translation_mean_m: NDArray[np.float64]  # (3,)
    translation_std_m: float
    rotation_mean_deg: float
    rotation_std_deg: float
    num_calibrations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "translation_mean_m": self.translation_mean_m.tolist(),
            "translation_std_m": self.translation_std_m,
            "rotation_mean_deg": self.rotation_mean_deg,
            "rotation_std_deg": self.rotation_std_deg,
            "num_calibrations": self.num_calibrations,
        }


def compute_repeatability_metrics(
    transforms: List[NDArray[np.float64]],
) -> RepeatabilityMetrics:
    """Compute repeatability metrics from multiple calibration transforms.
    
    Args:
        transforms: List of 4x4 transformation matrices.
        
    Returns:
        RepeatabilityMetrics summarizing variation across calibrations.
    """
    if len(transforms) < 2:
        raise ValueError("Need at least 2 transforms for repeatability analysis")
    
    # Extract translations and rotations
    translations = []
    rotations = []
    
    for T in transforms:
        R, t = decompose_transform(T)
        translations.append(t)
        rotations.append(R)
    
    translations = np.array(translations)  # (N, 3)
    
    # Translation statistics
    translation_mean = np.mean(translations, axis=0)
    translation_deviations = np.linalg.norm(translations - translation_mean, axis=1)
    translation_std = float(np.std(translation_deviations))
    
    # Rotation statistics (using angle between rotation matrices)
    angles_deg = []
    R_mean = rotations[0]  # Use first as reference
    
    for R in rotations:
        # Compute relative rotation
        R_rel = R @ R_mean.T
        angle_rad = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        angles_deg.append(np.degrees(angle_rad))
    
    rotation_mean = float(np.mean(angles_deg))
    rotation_std = float(np.std(angles_deg))
    
    return RepeatabilityMetrics(
        translation_mean_m=translation_mean,
        translation_std_m=translation_std,
        rotation_mean_deg=rotation_mean,
        rotation_std_deg=rotation_std,
        num_calibrations=len(transforms),
    )


@dataclass
class ComparisonResult:
    """Result of comparing two calibrations.
    
    Attributes:
        translation_diff_m: Translation difference (3,).
        translation_diff_norm_m: Translation difference magnitude (m).
        rotation_diff_deg: Rotation difference (degrees).
    """
    
    translation_diff_m: NDArray[np.float64]
    translation_diff_norm_m: float
    rotation_diff_deg: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "translation_diff_m": self.translation_diff_m.tolist(),
            "translation_diff_norm_m": self.translation_diff_norm_m,
            "rotation_diff_deg": self.rotation_diff_deg,
        }


def compare_calibrations(
    T1: NDArray[np.float64],
    T2: NDArray[np.float64],
) -> ComparisonResult:
    """Compare two calibration transforms.
    
    Args:
        T1: First 4x4 transformation matrix.
        T2: Second 4x4 transformation matrix.
        
    Returns:
        ComparisonResult with differences.
    """
    R1, t1 = decompose_transform(T1)
    R2, t2 = decompose_transform(T2)
    
    # Translation difference
    translation_diff = t2 - t1
    translation_diff_norm = float(np.linalg.norm(translation_diff))
    
    # Rotation difference (geodesic distance on SO(3))
    R_rel = R2 @ R1.T
    angle_rad = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
    rotation_diff_deg = float(np.degrees(angle_rad))
    
    return ComparisonResult(
        translation_diff_m=translation_diff,
        translation_diff_norm_m=translation_diff_norm,
        rotation_diff_deg=rotation_diff_deg,
    )
