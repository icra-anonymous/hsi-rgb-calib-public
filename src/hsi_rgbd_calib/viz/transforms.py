"""Centralized transform utilities for visualization.

This module provides the CalibratedRig class which is the single source
of truth for all coordinate transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    compute_transform_pattern_to_linescan,
    compute_scan_line_in_pattern,
)


@dataclass
class CalibratedRig:
    """Single source of truth for all rig transforms.
    
    This class centralizes all transform computations to prevent
    double-inversion bugs across visualization modules.
    
    The primary input is the artifact's T_oakrgb_hsi which transforms
    points FROM HSI TO OAK-RGB frame.
    
    Attributes:
        T_oakrgb_hsi: 4x4 transform matrix (HSI→OAK-RGB).
    """
    
    T_oakrgb_hsi: NDArray[np.float64]
    
    def __post_init__(self):
        """Validate transform matrix."""
        if self.T_oakrgb_hsi.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {self.T_oakrgb_hsi.shape}")
    
    @cached_property
    def R_hsi_oakrgb(self) -> NDArray[np.float64]:
        """Rotation: OAK-RGB → HSI (inverse of artifact rotation)."""
        return self.T_oakrgb_hsi[:3, :3].T
    
    @cached_property
    def t_hsi_oakrgb(self) -> NDArray[np.float64]:
        """Translation: OAK-RGB → HSI (inverse of artifact translation)."""
        return -self.R_hsi_oakrgb @ self.T_oakrgb_hsi[:3, 3]
    
    @cached_property
    def T_hsi_oakrgb(self) -> NDArray[np.float64]:
        """4x4 transform: OAK-RGB → HSI."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R_hsi_oakrgb
        T[:3, 3] = self.t_hsi_oakrgb
        return T
    
    def pattern_to_hsi(
        self,
        R_frame_pattern: NDArray[np.float64],
        T_frame_pattern: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute composed pattern→HSI transform for a view.
        
        Args:
            R_frame_pattern: 3x3 rotation pattern→frame-camera (from pose).
            T_frame_pattern: 3-element translation pattern→frame (from pose).
            
        Returns:
            Tuple of (R0, T0) - the pattern→HSI transform.
        """
        return compute_transform_pattern_to_linescan(
            R_frame_pattern, T_frame_pattern,
            self.R_hsi_oakrgb, self.t_hsi_oakrgb,
        )
    
    def get_scan_line(
        self,
        R_frame_pattern: NDArray[np.float64],
        T_frame_pattern: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        """Compute scan line in pattern coordinates for a view.
        
        Args:
            R_frame_pattern: 3x3 rotation pattern→frame-camera.
            T_frame_pattern: 3-element translation pattern→frame.
            
        Returns:
            Line coefficients (a, b, c) where aX + bY + c = 0.
        """
        R0, T0 = self.pattern_to_hsi(R_frame_pattern, T_frame_pattern)
        return compute_scan_line_in_pattern(R0, T0)
    
    def hsi_origin_in_oakrgb(self) -> NDArray[np.float64]:
        """Get HSI camera origin in OAK-RGB frame."""
        # HSI origin (0,0,0) in HSI coords → transform to OAK-RGB
        return self.T_oakrgb_hsi[:3, 3]
    
    def oakrgb_origin_in_hsi(self) -> NDArray[np.float64]:
        """Get OAK-RGB camera origin in HSI frame."""
        return self.t_hsi_oakrgb


def check_view_validity(
    R0: NDArray[np.float64],
    T0: NDArray[np.float64],
    P_pattern: NDArray[np.float64],
) -> dict:
    """Check physical validity of a view.
    
    Performs chirality checks to ensure the pattern is in front of
    the camera and poses are physically valid.
    
    Args:
        R0: 3x3 pattern→HSI rotation matrix.
        T0: 3-element pattern→HSI translation.
        P_pattern: Nx3 array of pattern points.
        
    Returns:
        Dictionary with validity checks:
        - centroid_in_front: Pattern centroid has positive depth.
        - det_R0_positive: Rotation has positive determinant.
        - points_visible: At least 4/6 points have positive depth.
    """
    # Transform points to HSI frame
    depths = []
    for p in P_pattern:
        p_hsi = R0 @ p + T0
        depths.append(p_hsi[2])
    
    centroid = P_pattern.mean(axis=0)
    centroid_hsi = R0 @ centroid + T0
    
    return {
        "centroid_in_front": centroid_hsi[2] > 0,
        "det_R0_positive": np.linalg.det(R0) > 0,
        "points_visible": sum(d > 0 for d in depths) >= 4,
    }
