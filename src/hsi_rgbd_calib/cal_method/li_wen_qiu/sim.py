"""Simulation utilities for Li-Wen-Qiu calibration.

This module provides functions to generate synthetic calibration data
for testing and validation of the Li-Wen-Qiu calibration method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.boards.li_wen_qiu_pattern import (
    LiWenQiuPattern,
    get_default_li_wen_qiu_pattern,
)
from hsi_rgbd_calib.boards.geometry import intersect_lines_2d
from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    project_to_linescan,
    compute_transform_pattern_to_linescan,
    compute_scan_line_in_pattern,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.backend import ViewObservation
from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GroundTruth:
    """Ground truth calibration parameters.
    
    Attributes:
        f: Focal length of line-scan camera.
        v0: Principal point.
        k: Radial distortion coefficient.
        R: 3x3 rotation matrix (frame-to-line-scan).
        T: 3-element translation vector.
    """
    f: float
    v0: float
    k: float
    R: NDArray[np.float64]
    T: NDArray[np.float64]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "f": self.f,
            "v0": self.v0,
            "k": self.k,
            "R": self.R.tolist(),
            "T": self.T.tolist(),
        }


@dataclass
class NoiseConfig:
    """Configuration for simulation noise.
    
    Attributes:
        sigma_v: Standard deviation of pixel noise (pixels).
        sigma_pose_t: Standard deviation of pose translation noise (meters).
        sigma_pose_r: Standard deviation of pose rotation noise (radians).
    """
    sigma_v: float = 0.2
    sigma_pose_t: float = 0.0
    sigma_pose_r: float = 0.0


@dataclass
class SimulationResult:
    """Result of simulation.
    
    Attributes:
        views: List of generated ViewObservation objects.
        ground_truth: Ground truth parameters.
        pattern: Pattern used for simulation.
        noise_config: Noise configuration used.
    """
    views: List[ViewObservation]
    ground_truth: GroundTruth
    pattern: LiWenQiuPattern
    noise_config: NoiseConfig


def generate_random_pose(
    distance_range: Tuple[float, float] = (0.3, 0.8),
    angle_range: Tuple[float, float] = (-20, 20),
    seed: Optional[int] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a random pattern-to-frame-camera pose.
    
    The pattern is placed in front of the camera at a random distance
    and orientation.
    
    Args:
        distance_range: (min, max) distance from camera to pattern center.
        angle_range: (min, max) tilt angles in degrees.
        seed: Optional random seed.
        
    Returns:
        Tuple of (R, T) where R is 3x3 rotation and T is translation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random distance
    distance = np.random.uniform(*distance_range)
    
    # Random rotation angles (in degrees)
    rx = np.random.uniform(*angle_range)
    ry = np.random.uniform(*angle_range)
    rz = np.random.uniform(-10, 10)  # Smaller in-plane rotation
    
    # Create rotation matrix
    R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    
    # Translation: pattern center is at (0, 0, distance) in camera frame
    # with some lateral offset
    tx = np.random.uniform(-0.1, 0.1)
    ty = np.random.uniform(-0.1, 0.1)
    T = np.array([tx, ty, distance])
    
    return R, T


def generate_poses(
    n_views: int,
    distance_range: Tuple[float, float] = (0.3, 0.8),
    angle_range: Tuple[float, float] = (-20, 20),
    seed: Optional[int] = None,
) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Generate multiple random poses.
    
    Args:
        n_views: Number of poses to generate.
        distance_range: Distance range for each pose.
        angle_range: Angle range for each pose.
        seed: Optional random seed.
        
    Returns:
        List of (R, T) tuples.
    """
    if seed is not None:
        np.random.seed(seed)
    
    poses = []
    for i in range(n_views):
        R, T = generate_random_pose(distance_range, angle_range)
        poses.append((R, T))
    
    return poses


def simulate_view(
    R_frame_pattern: NDArray[np.float64],
    T_frame_pattern: NDArray[np.float64],
    ground_truth: GroundTruth,
    pattern: LiWenQiuPattern,
    noise_config: NoiseConfig,
) -> ViewObservation:
    """Simulate a single view observation.
    
    Algorithm:
    1. Compute pattern-to-line-scan transform (Eq 27)
    2. Compute scan line in pattern plane (Eq 30)
    3. Intersect with feature lines to get P_i
    4. Project P_i to line-scan coordinates (Eq 4)
    5. Add measurement noise
    
    Args:
        R_frame_pattern: Pattern-to-frame rotation.
        T_frame_pattern: Pattern-to-frame translation.
        ground_truth: Ground truth calibration parameters.
        pattern: Li-Wen-Qiu pattern model.
        noise_config: Noise configuration.
        
    Returns:
        ViewObservation with simulated measurements.
    """
    # Step 1: Compute pattern-to-line-scan transform
    R0, T0 = compute_transform_pattern_to_linescan(
        R_frame_pattern, T_frame_pattern,
        ground_truth.R, ground_truth.T
    )
    
    # Step 2: Compute scan line in pattern plane
    scan_line = compute_scan_line_in_pattern(R0, T0)
    
    # Step 3 & 4: Intersect with feature lines and project
    v_observations = np.zeros(6, dtype=np.float64)
    
    for i in range(6):
        feature_line = pattern.feature_lines[i]
        
        pt = intersect_lines_2d(scan_line, feature_line)
        if pt is None:
            raise ValueError(f"Scan line parallel to feature line L{i+1}")
        
        P_i = np.array([pt[0], pt[1], 0.0])
        
        # Project using ground truth parameters
        v = project_to_linescan(
            P_i, R0, T0,
            ground_truth.f, ground_truth.v0, ground_truth.k
        )
        
        # Step 5: Add noise
        if noise_config.sigma_v > 0:
            v += np.random.normal(0, noise_config.sigma_v)
        
        v_observations[i] = v
    
    # Apply pose noise if specified
    R_noisy = R_frame_pattern.copy()
    T_noisy = T_frame_pattern.copy()
    
    if noise_config.sigma_pose_r > 0:
        noise_rvec = np.random.normal(0, noise_config.sigma_pose_r, 3)
        R_noise = Rotation.from_rotvec(noise_rvec).as_matrix()
        R_noisy = R_noisy @ R_noise
    
    if noise_config.sigma_pose_t > 0:
        T_noisy += np.random.normal(0, noise_config.sigma_pose_t, 3)
    
    return ViewObservation(
        R_frame_pattern=R_noisy,
        T_frame_pattern=T_noisy,
        v_observations=v_observations,
    )


def simulate_views(
    n_views: int,
    ground_truth: Optional[GroundTruth] = None,
    pattern: Optional[LiWenQiuPattern] = None,
    noise_config: Optional[NoiseConfig] = None,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Generate complete simulation with multiple views.
    
    Args:
        n_views: Number of views to generate.
        ground_truth: Ground truth parameters (default: typical values).
        pattern: Pattern model (default: Li-Wen-Qiu default).
        noise_config: Noise configuration (default: 0.2 px noise).
        seed: Random seed for reproducibility.
        
    Returns:
        SimulationResult with generated data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use defaults
    if pattern is None:
        pattern = get_default_li_wen_qiu_pattern()
    
    if noise_config is None:
        noise_config = NoiseConfig()
    
    if ground_truth is None:
        ground_truth = get_default_ground_truth()
    
    # Generate poses
    poses = generate_poses(n_views, seed=seed)
    
    # Simulate each view
    views = []
    for i, (R, T) in enumerate(poses):
        try:
            view = simulate_view(R, T, ground_truth, pattern, noise_config)
            view.view_id = f"sim_view_{i:03d}"
            views.append(view)
        except ValueError as e:
            logger.warning(f"Skipping view {i}: {e}")
    
    return SimulationResult(
        views=views,
        ground_truth=ground_truth,
        pattern=pattern,
        noise_config=noise_config,
    )


def get_default_ground_truth() -> GroundTruth:
    """Get default ground truth parameters for simulation.
    
    Returns:
        GroundTruth with typical values.
    """
    # Typical line-scan camera parameters
    f = 1150.0   # Focal length in pixels
    v0 = 640.0   # Principal point (middle of 1280-pixel sensor)
    k = -1e-7    # Small radial distortion
    
    # Small rotation and translation between cameras
    R = Rotation.from_euler('xyz', [2, -3, 1], degrees=True).as_matrix()
    T = np.array([0.05, 0.01, 0.08])  # 5cm, 1cm, 8cm offset
    
    return GroundTruth(f=f, v0=v0, k=k, R=R, T=T)


def compute_estimation_errors(
    estimated: Dict[str, Any],
    ground_truth: GroundTruth,
) -> Dict[str, float]:
    """Compute errors between estimated and ground truth parameters.
    
    Args:
        estimated: Dictionary with estimated parameters.
        ground_truth: Ground truth parameters.
        
    Returns:
        Dictionary with error metrics.
    """
    # Rotation error
    R_est = np.array(estimated["R"])
    R_gt = ground_truth.R
    R_diff = R_est @ R_gt.T
    rotation_error_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    rotation_error_deg = np.degrees(rotation_error_rad)
    
    # Translation error
    T_est = np.array(estimated["T"])
    T_gt = ground_truth.T
    translation_error_m = np.linalg.norm(T_est - T_gt)
    
    # Focal length error
    f_est = estimated["f"]
    f_error_rel = abs(f_est - ground_truth.f) / ground_truth.f
    
    # Principal point error
    v0_est = estimated["v0"]
    v0_error_px = abs(v0_est - ground_truth.v0)
    
    # Distortion error
    k_est = estimated.get("k", 0.0)
    k_error = abs(k_est - ground_truth.k)
    
    return {
        "rotation_error_rad": rotation_error_rad,
        "rotation_error_deg": rotation_error_deg,
        "translation_error_m": translation_error_m,
        "f_error_relative": f_error_rel,
        "v0_error_px": v0_error_px,
        "k_error": k_error,
    }
