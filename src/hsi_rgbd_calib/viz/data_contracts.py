"""Data contracts for calibration visualization.

This module defines the data structures used to pass calibration results
to the visualization functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class GroundTruthViz:
    """Ground truth data for simulation comparison.
    
    Attributes:
        R_true: True rotation matrix (frame→HSI).
        T_true: True translation vector.
        f_true: True focal length.
        v0_true: True principal point.
        k_true: True distortion coefficient.
    """
    R_true: NDArray[np.float64]
    T_true: NDArray[np.float64]
    f_true: float
    v0_true: float
    k_true: float = 0.0


@dataclass
class ViewVizData:
    """Per-view visualization data.
    
    Attributes:
        view_id: Unique identifier for this view.
        R_frame_pattern: 3x3 rotation pattern→frame camera.
        T_frame_pattern: 3-element translation pattern→frame (meters).
        R0: 3x3 composed rotation pattern→HSI.
        T0: 3-element composed translation pattern→HSI (meters).
        scan_line: (a, b, c) coefficients of scan line in pattern plane.
        v_observed: 6-element array of observed pixel coordinates.
        v_init: 6-element array of initial predictions (after closed-form).
        v_final: 6-element array of final predictions (after refinement).
        P_pattern_init: 6x3 array of initial recovered pattern points.
        P_pattern_final: 6x3 array of final recovered pattern points.
        residual_rmse: RMS residual for this view (pixels).
        frame_image_path: Optional path to frame RGB image.
        image_association_verified: Whether image association has been verified.
    """
    
    view_id: str
    R_frame_pattern: NDArray[np.float64]
    T_frame_pattern: NDArray[np.float64]
    R0: NDArray[np.float64]
    T0: NDArray[np.float64]
    scan_line: Tuple[float, float, float]
    v_observed: NDArray[np.float64]
    v_init: NDArray[np.float64]
    v_final: NDArray[np.float64]
    P_pattern_init: NDArray[np.float64]
    P_pattern_final: NDArray[np.float64]
    residual_rmse: float
    frame_image_path: Optional[Path] = None
    image_association_verified: bool = False


@dataclass
class VisualizationData:
    """Complete visualization data package.
    
    Attributes:
        wp1: Pattern dimension 1 (meters).
        wp2: Pattern dimension 2 (meters).
        pattern_lines: 6 feature lines [(a, b, c), ...].
        T_oakrgb_hsi: 4x4 transform HSI→OAK-RGB (as in artifact).
        f: HSI focal length (pixels).
        v0: HSI principal point (pixels).
        k: HSI distortion coefficient.
        views: Per-view calibration data.
        frame_K: Optional frame camera intrinsic matrix (3x3).
        frame_dist: Optional frame camera distortion (5,).
        frame_size: Optional frame image size (width, height).
        cost_history: Optional optimization cost history.
        gt: Optional ground truth for simulation.
    """
    
    wp1: float
    wp2: float
    pattern_lines: List[Tuple[float, float, float]]
    T_oakrgb_hsi: NDArray[np.float64]
    f: float
    v0: float
    k: float
    views: List[ViewVizData]
    
    frame_K: Optional[NDArray[np.float64]] = None
    frame_dist: Optional[NDArray[np.float64]] = None
    frame_size: Optional[Tuple[int, int]] = None
    cost_history: Optional[List[float]] = None
    gt: Optional[GroundTruthViz] = None
    
    @classmethod
    def from_calibration_result(
        cls,
        result,  # CalibrationResult
        pattern,  # LiWenQiuPattern
        gt: Optional[GroundTruthViz] = None,
    ) -> "VisualizationData":
        """Create VisualizationData from a CalibrationResult.
        
        Args:
            result: CalibrationResult from the backend.
            pattern: LiWenQiuPattern used for calibration.
            gt: Optional ground truth for simulation.
            
        Returns:
            VisualizationData ready for plotting.
        """
        views = []
        for vr in result.per_view:
            views.append(ViewVizData(
                view_id=vr.view_id,
                R_frame_pattern=vr.R_frame_pattern,
                T_frame_pattern=vr.T_frame_pattern,
                R0=vr.R0,
                T0=vr.T0,
                scan_line=vr.scan_line,
                v_observed=vr.v_observed,
                v_init=vr.v_init,
                v_final=vr.v_final,
                P_pattern_init=vr.P_pattern_init,
                P_pattern_final=vr.P_pattern_final,
                residual_rmse=vr.residual_rmse,
            ))
        
        return cls(
            wp1=pattern.wp1,
            wp2=pattern.wp2,
            pattern_lines=pattern.feature_lines,
            T_oakrgb_hsi=result.T_oakrgb_hsi,
            f=result.hsi_intrinsics.focal_length_slit,
            v0=result.hsi_intrinsics.principal_point_u0,
            k=result.hsi_intrinsics.distortion_coeffs[0] if result.hsi_intrinsics.distortion_coeffs else 0.0,
            views=views,
            cost_history=result.cost_history,
            gt=gt,
        )
