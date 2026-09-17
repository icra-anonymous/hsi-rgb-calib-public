"""Calibration method interface.

This module defines the interface for the calibration method, which estimates
the extrinsic transformation between HSI and RGB cameras, as well as
HSI slit intrinsics.

The implementation adopts the Li-Wen-Qiu method for line-scan + frame camera
calibration. Two backends are available:
- StubBackend: Loads precomputed results from a file
- PythonBackend: Placeholder implementation with TODOs for future development
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from hsi_rgbd_calib.common.logging import get_logger
from hsi_rgbd_calib.io.session import SessionData

logger = get_logger(__name__)


class CalibrationBackend(Enum):
    """Available calibration backends."""
    
    STUB = "stub"
    PYTHON = "python"


@dataclass
class CalibrationConfig:
    """Configuration for the calibration method.
    
    Attributes:
        backend: Which backend to use ("stub" or "python").
        method_name: Name of the calibration method.
        max_iterations: Maximum optimization iterations.
        convergence_threshold: Convergence threshold for optimization.
        use_ransac: Whether to use RANSAC for outlier rejection.
        ransac_threshold_px: RANSAC inlier threshold in pixels.
        min_correspondences: Minimum number of correspondences required.
        verbose: Whether to print verbose output.
    """
    
    backend: str = "stub"
    method_name: str = "li_wen_qiu"
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    use_ransac: bool = True
    ransac_threshold_px: float = 2.0
    min_correspondences: int = 20
    verbose: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationConfig":
        """Create from dictionary."""
        return cls(
            backend=data.get("backend", "stub"),
            method_name=data.get("method_name", "li_wen_qiu"),
            max_iterations=data.get("max_iterations", 100),
            convergence_threshold=data.get("convergence_threshold", 1e-6),
            use_ransac=data.get("use_ransac", True),
            ransac_threshold_px=data.get("ransac_threshold_px", 2.0),
            min_correspondences=data.get("min_correspondences", 20),
            verbose=data.get("verbose", True),
        )
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "CalibrationConfig":
        """Load from a YAML file."""
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        return cls.from_dict(data)


@dataclass
class HsiSlitIntrinsicsResult:
    """HSI slit intrinsics estimation result.
    
    Attributes:
        focal_length_slit: Focal length along slit direction (pixels).
        principal_point_u0: Principal point along slit (pixels).
        slit_width: Number of pixels along slit.
        distortion_coeffs: Distortion coefficients (if estimated).
    """
    
    focal_length_slit: float
    principal_point_u0: float
    slit_width: int
    distortion_coeffs: List[float] = field(default_factory=list)


@dataclass
class ViewResult:
    """Per-view calibration data for visualization and debugging.
    
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
    """
    
    view_id: str
    R_frame_pattern: NDArray[np.float64]
    T_frame_pattern: NDArray[np.float64]
    R0: NDArray[np.float64]
    T0: NDArray[np.float64]
    scan_line: tuple  # (a, b, c)
    v_observed: NDArray[np.float64]
    v_init: NDArray[np.float64]
    v_final: NDArray[np.float64]
    P_pattern_init: NDArray[np.float64]
    P_pattern_final: NDArray[np.float64]
    residual_rmse: float


@dataclass
class CalibrationResult:
    """Result of the calibration process.
    
    Attributes:
        T_oakrgb_hsi: 4x4 extrinsic transformation from HSI to OAK RGB.
        hsi_intrinsics: Estimated HSI slit intrinsics.
        reprojection_error_rmse: RMS reprojection error in pixels.
        reprojection_error_median: Median reprojection error in pixels.
        reprojection_error_max: Maximum reprojection error in pixels.
        num_correspondences: Number of point correspondences used.
        num_inliers: Number of inliers after RANSAC (if used).
        translation_std_m: Standard deviation of translation estimate (meters).
        rotation_std_deg: Standard deviation of rotation estimate (degrees).
        method: Method name used.
        success: Whether calibration succeeded.
        message: Status message or error description.
        per_view: Per-view calibration data for visualization.
        cost_history: Optimization cost history for visualization.
    """
    
    T_oakrgb_hsi: NDArray[np.float64]
    hsi_intrinsics: HsiSlitIntrinsicsResult
    reprojection_error_rmse: float
    reprojection_error_median: float
    reprojection_error_max: float
    num_correspondences: int
    num_inliers: int
    translation_std_m: Optional[float] = None
    rotation_std_deg: Optional[float] = None
    method: str = "li_wen_qiu"
    success: bool = True
    message: str = "Calibration completed successfully"
    per_view: List[ViewResult] = field(default_factory=list)
    cost_history: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "T_oakrgb_hsi": self.T_oakrgb_hsi.tolist(),
            "hsi_intrinsics": {
                "focal_length_slit": self.hsi_intrinsics.focal_length_slit,
                "principal_point_u0": self.hsi_intrinsics.principal_point_u0,
                "slit_width": self.hsi_intrinsics.slit_width,
                "distortion_coeffs": self.hsi_intrinsics.distortion_coeffs,
            },
            "reprojection_error_rmse": self.reprojection_error_rmse,
            "reprojection_error_median": self.reprojection_error_median,
            "reprojection_error_max": self.reprojection_error_max,
            "num_correspondences": self.num_correspondences,
            "num_inliers": self.num_inliers,
            "translation_std_m": self.translation_std_m,
            "rotation_std_deg": self.rotation_std_deg,
            "method": self.method,
            "success": self.success,
            "message": self.message,
        }


def estimate_calibration(
    session: SessionData,
    config: CalibrationConfig,
) -> CalibrationResult:
    """Estimate calibration using the configured backend.
    
    This is the main entry point for the calibration method. It dispatches
    to either the stub or Python backend based on the configuration.
    
    Args:
        session: Loaded session data.
        config: Calibration configuration.
        
    Returns:
        CalibrationResult containing the estimated calibration.
        
    Raises:
        ValueError: If the backend is not recognized or calibration fails.
    """
    backend = config.backend.lower()
    
    if backend == "stub":
        from hsi_rgbd_calib.cal_method.stub_backend import StubBackend
        backend_impl = StubBackend()
    elif backend == "python":
        from hsi_rgbd_calib.cal_method.python_backend import PythonBackend
        backend_impl = PythonBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    logger.info(f"Running calibration with {backend} backend")
    return backend_impl.estimate(session, config)
