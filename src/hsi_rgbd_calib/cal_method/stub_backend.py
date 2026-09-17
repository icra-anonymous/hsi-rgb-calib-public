"""Stub backend for calibration method.

This backend loads precomputed calibration results from a file,
useful for testing, development, and when using external calibration tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml

from hsi_rgbd_calib.cal_method.interface import (
    CalibrationResult,
    CalibrationConfig,
    HsiSlitIntrinsicsResult,
)
from hsi_rgbd_calib.io.session import SessionData
from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


class StubBackend:
    """Stub backend that loads precomputed calibration results.
    
    This backend looks for a precomputed result file in the session's
    cal_method directory (e.g., cal_method_output.yaml).
    
    Example usage:
        >>> backend = StubBackend()
        >>> result = backend.estimate(session, config)
    """
    
    def estimate(
        self,
        session: SessionData,
        config: CalibrationConfig,
    ) -> CalibrationResult:
        """Load calibration result from precomputed file.
        
        Args:
            session: Loaded session data.
            config: Calibration configuration (not used for stub).
            
        Returns:
            CalibrationResult loaded from the precomputed file.
            
        Raises:
            ValueError: If no precomputed results are available.
        """
        if not session.has_precomputed_calibration:
            raise ValueError(
                "No precomputed calibration results found in session. "
                "Stub backend requires cal_method/cal_method_output.yaml"
            )
        
        logger.info("Loading precomputed calibration from stub backend")
        return load_precomputed_results(session.cal_method_output)


def load_precomputed_results(data: Dict[str, Any]) -> CalibrationResult:
    """Parse precomputed calibration results from a dictionary.
    
    Expected format:
    ```yaml
    T_oakrgb_hsi:
      - [1.0, 0.0, 0.0, 0.05]
      - [0.0, 1.0, 0.0, -0.02]
      - [0.0, 0.0, 1.0, 0.1]
      - [0.0, 0.0, 0.0, 1.0]
    
    hsi_intrinsics:
      focal_length_slit: 1200.0
      principal_point_u0: 640.0
      slit_width: 1280
      distortion_coeffs: []
    
    metrics:
      reprojection_error_rmse: 0.45
      reprojection_error_median: 0.38
      reprojection_error_max: 1.23
      num_correspondences: 150
      num_inliers: 145
    
    uncertainty:
      translation_std_m: 0.002
      rotation_std_deg: 0.15
    
    method: li_wen_qiu
    success: true
    message: "Precomputed calibration loaded successfully"
    ```
    
    Args:
        data: Dictionary with precomputed results.
        
    Returns:
        Parsed CalibrationResult.
    """
    # Parse extrinsic matrix
    T_data = data.get("T_oakrgb_hsi")
    if T_data is None:
        raise ValueError("Missing 'T_oakrgb_hsi' in precomputed results")
    
    T_oakrgb_hsi = np.array(T_data, dtype=np.float64)
    
    if T_oakrgb_hsi.shape != (4, 4):
        raise ValueError(f"T_oakrgb_hsi must be 4x4, got {T_oakrgb_hsi.shape}")
    
    # Parse HSI intrinsics
    hsi_data = data.get("hsi_intrinsics", {})
    hsi_intrinsics = HsiSlitIntrinsicsResult(
        focal_length_slit=float(hsi_data.get("focal_length_slit", 1000.0)),
        principal_point_u0=float(hsi_data.get("principal_point_u0", 640.0)),
        slit_width=int(hsi_data.get("slit_width", 1280)),
        distortion_coeffs=hsi_data.get("distortion_coeffs", []),
    )
    
    # Parse metrics
    metrics = data.get("metrics", {})
    reprojection_error_rmse = float(metrics.get("reprojection_error_rmse", 0.5))
    reprojection_error_median = float(metrics.get("reprojection_error_median", 0.4))
    reprojection_error_max = float(metrics.get("reprojection_error_max", 1.5))
    num_correspondences = int(metrics.get("num_correspondences", 100))
    num_inliers = int(metrics.get("num_inliers", 95))
    
    # Parse uncertainty
    uncertainty = data.get("uncertainty", {})
    translation_std_m = uncertainty.get("translation_std_m")
    rotation_std_deg = uncertainty.get("rotation_std_deg")
    
    return CalibrationResult(
        T_oakrgb_hsi=T_oakrgb_hsi,
        hsi_intrinsics=hsi_intrinsics,
        reprojection_error_rmse=reprojection_error_rmse,
        reprojection_error_median=reprojection_error_median,
        reprojection_error_max=reprojection_error_max,
        num_correspondences=num_correspondences,
        num_inliers=num_inliers,
        translation_std_m=translation_std_m,
        rotation_std_deg=rotation_std_deg,
        method=data.get("method", "li_wen_qiu"),
        success=data.get("success", True),
        message=data.get("message", "Loaded from precomputed results"),
    )


def save_precomputed_results(
    result: CalibrationResult,
    output_path: Path | str,
) -> Path:
    """Save calibration results to a YAML file for later use with stub backend.
    
    Args:
        result: CalibrationResult to save.
        output_path: Path to write the YAML file.
        
    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "T_oakrgb_hsi": result.T_oakrgb_hsi.tolist(),
        "hsi_intrinsics": {
            "focal_length_slit": result.hsi_intrinsics.focal_length_slit,
            "principal_point_u0": result.hsi_intrinsics.principal_point_u0,
            "slit_width": result.hsi_intrinsics.slit_width,
            "distortion_coeffs": result.hsi_intrinsics.distortion_coeffs,
        },
        "metrics": {
            "reprojection_error_rmse": result.reprojection_error_rmse,
            "reprojection_error_median": result.reprojection_error_median,
            "reprojection_error_max": result.reprojection_error_max,
            "num_correspondences": result.num_correspondences,
            "num_inliers": result.num_inliers,
        },
        "uncertainty": {
            "translation_std_m": result.translation_std_m,
            "rotation_std_deg": result.rotation_std_deg,
        },
        "method": result.method,
        "success": result.success,
        "message": result.message,
    }
    
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved precomputed results to {output_path}")
    return output_path
