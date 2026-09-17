"""Reprojection error computation.

This module provides functions for computing reprojection errors
for both frame cameras and HSI line-scan cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReprojectionMetrics:
    """Reprojection error metrics.
    
    Attributes:
        rmse: Root mean square error.
        median: Median absolute error.
        max: Maximum error.
        std: Standard deviation.
        num_points: Number of points used.
    """
    
    rmse: float
    median: float
    max: float
    std: float
    num_points: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rmse": self.rmse,
            "median": self.median,
            "max": self.max,
            "std": self.std,
            "num_points": self.num_points,
        }


def project_points_to_camera(
    points_3d: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    T_camera_world: NDArray[np.float64],
    dist_coeffs: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Project 3D points to a frame camera.
    
    Args:
        points_3d: 3D points in world coordinates (N, 3).
        camera_matrix: 3x3 camera intrinsic matrix K.
        T_camera_world: 4x4 transform from world to camera.
        dist_coeffs: Optional distortion coefficients.
        
    Returns:
        Projected 2D points (N, 2).
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    T_camera_world = np.asarray(T_camera_world, dtype=np.float64)
    
    N = len(points_3d)
    
    # Transform to camera frame
    R = T_camera_world[:3, :3]
    t = T_camera_world[:3, 3]
    points_cam = (R @ points_3d.T).T + t
    
    # Project to image plane
    points_2d = np.zeros((N, 2), dtype=np.float64)
    
    for i in range(N):
        if abs(points_cam[i, 2]) < 1e-10:
            points_2d[i] = [np.nan, np.nan]
            continue
        
        x_norm = points_cam[i, 0] / points_cam[i, 2]
        y_norm = points_cam[i, 1] / points_cam[i, 2]
        
        # Apply distortion if provided
        if dist_coeffs is not None and len(dist_coeffs) > 0:
            x_norm, y_norm = _apply_distortion(x_norm, y_norm, dist_coeffs)
        
        # Apply camera matrix
        u = camera_matrix[0, 0] * x_norm + camera_matrix[0, 2]
        v = camera_matrix[1, 1] * y_norm + camera_matrix[1, 2]
        
        points_2d[i] = [u, v]
    
    return points_2d


def _apply_distortion(
    x: float, y: float, 
    coeffs: NDArray[np.float64]
) -> Tuple[float, float]:
    """Apply radial-tangential distortion model.
    
    Args:
        x, y: Normalized image coordinates.
        coeffs: Distortion coefficients [k1, k2, p1, p2, k3, ...].
        
    Returns:
        Distorted coordinates.
    """
    k1 = coeffs[0] if len(coeffs) > 0 else 0
    k2 = coeffs[1] if len(coeffs) > 1 else 0
    p1 = coeffs[2] if len(coeffs) > 2 else 0
    p2 = coeffs[3] if len(coeffs) > 3 else 0
    k3 = coeffs[4] if len(coeffs) > 4 else 0
    
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    
    # Radial distortion
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    
    # Tangential distortion
    x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    
    return x_dist, y_dist


def project_points_to_slit(
    points_3d: NDArray[np.float64],
    T_hsi_world: NDArray[np.float64],
    focal_length_slit: float,
    principal_point_u0: float,
) -> NDArray[np.float64]:
    """Project 3D points to an HSI slit.
    
    For a pushbroom camera, this produces only a 1D coordinate
    along the slit direction.
    
    Args:
        points_3d: 3D points in world coordinates (N, 3).
        T_hsi_world: 4x4 transform from world to HSI frame.
        focal_length_slit: Focal length along slit.
        principal_point_u0: Principal point offset.
        
    Returns:
        Slit coordinates (N,).
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    T_hsi_world = np.asarray(T_hsi_world, dtype=np.float64)
    
    N = len(points_3d)
    
    # Transform to HSI frame
    R = T_hsi_world[:3, :3]
    t = T_hsi_world[:3, 3]
    points_hsi = (R @ points_3d.T).T + t
    
    # Project to slit (x/z ratio)
    slit_coords = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        if abs(points_hsi[i, 2]) < 1e-10:
            slit_coords[i] = np.nan
            continue
        
        u = focal_length_slit * (points_hsi[i, 0] / points_hsi[i, 2]) + principal_point_u0
        slit_coords[i] = u
    
    return slit_coords


def compute_reprojection_error(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute reprojection error between observed and predicted points.
    
    Args:
        observed: Observed points (N,) for 1D or (N, 2) for 2D.
        predicted: Predicted points, same shape as observed.
        
    Returns:
        Error magnitudes (N,).
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    
    diff = observed - predicted
    
    if diff.ndim == 1:
        # 1D (slit) coordinates
        errors = np.abs(diff)
    else:
        # 2D coordinates
        errors = np.linalg.norm(diff, axis=1)
    
    return errors


def compute_reprojection_metrics(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
) -> ReprojectionMetrics:
    """Compute reprojection error metrics.
    
    Args:
        observed: Observed points.
        predicted: Predicted points.
        
    Returns:
        ReprojectionMetrics with error statistics.
    """
    errors = compute_reprojection_error(observed, predicted)
    
    # Filter out NaN values
    valid_errors = errors[~np.isnan(errors)]
    
    if len(valid_errors) == 0:
        return ReprojectionMetrics(
            rmse=np.nan,
            median=np.nan,
            max=np.nan,
            std=np.nan,
            num_points=0,
        )
    
    rmse = np.sqrt(np.mean(valid_errors ** 2))
    median = float(np.median(valid_errors))
    max_error = float(np.max(valid_errors))
    std = float(np.std(valid_errors))
    
    return ReprojectionMetrics(
        rmse=float(rmse),
        median=median,
        max=max_error,
        std=std,
        num_points=len(valid_errors),
    )
