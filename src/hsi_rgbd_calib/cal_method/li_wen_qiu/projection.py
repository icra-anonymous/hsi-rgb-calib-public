"""Line-scan camera projection model.

This module implements the line-scan camera projection model from
the Li-Wen-Qiu paper, including radial distortion.

Reference: Equation (4) in the paper.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def project_to_linescan(
    point_3d: NDArray[np.float64],
    R: NDArray[np.float64],
    T: NDArray[np.float64],
    f: float,
    v0: float,
    k: float = 0.0,
) -> float:
    """Project a 3D point to line-scan pixel coordinate.
    
    Implements Equation (4) from the Li-Wen-Qiu paper:
        s = (r21*X + r22*Y + r23*Z + t2) / (r31*X + r32*Y + r33*Z + t3)
        v = f * s * (1 + k*s^2) + v0
    
    Args:
        point_3d: 3D point in pattern coordinates (X, Y, Z).
        R: 3x3 rotation matrix from pattern to line-scan.
        T: 3-element translation vector.
        f: Focal length of line-scan camera.
        v0: Principal point offset.
        k: Radial distortion coefficient (default 0).
        
    Returns:
        Pixel coordinate v on the line-scan sensor.
    """
    X, Y, Z = point_3d
    
    # Extract rotation matrix elements
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
    t2, t3 = T[1], T[2]
    
    # Compute normalized coordinate s
    numerator = r21 * X + r22 * Y + r23 * Z + t2
    denominator = r31 * X + r32 * Y + r33 * Z + t3
    
    if abs(denominator) < 1e-12:
        return np.nan
    
    s = numerator / denominator
    
    # Apply distortion and compute pixel coordinate
    v = f * s * (1.0 + k * s * s) + v0
    
    return v


def project_to_linescan_batch(
    points_3d: NDArray[np.float64],
    R: NDArray[np.float64],
    T: NDArray[np.float64],
    f: float,
    v0: float,
    k: float = 0.0,
) -> NDArray[np.float64]:
    """Project multiple 3D points to line-scan coordinates.
    
    Vectorized version of project_to_linescan.
    
    Args:
        points_3d: Nx3 array of 3D points in pattern coordinates.
        R: 3x3 rotation matrix from pattern to line-scan.
        T: 3-element translation vector.
        f: Focal length of line-scan camera.
        v0: Principal point offset.
        k: Radial distortion coefficient.
        
    Returns:
        Array of N pixel coordinates.
    """
    N = len(points_3d)
    v = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        v[i] = project_to_linescan(points_3d[i], R, T, f, v0, k)
    
    return v


def compute_transform_pattern_to_linescan(
    R_frame_pattern: NDArray[np.float64],
    T_frame_pattern: NDArray[np.float64],
    R_linescan_frame: NDArray[np.float64],
    T_linescan_frame: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute pattern-to-line-scan transform from composed transforms.
    
    Implements Equation (27) from the paper:
        R0_j = R @ R_j
        T0_j = R @ T_j + T
    
    Where:
        R_j, T_j = pattern-to-frame-camera transform (for view j)
        R, T = frame-camera-to-line-scan extrinsics
        R0_j, T0_j = resulting pattern-to-line-scan transform
    
    Args:
        R_frame_pattern: 3x3 rotation from pattern to frame camera (R_j).
        T_frame_pattern: 3-element translation from pattern to frame (T_j).
        R_linescan_frame: 3x3 rotation from frame to line-scan (R).
        T_linescan_frame: 3-element translation from frame to line-scan (T).
        
    Returns:
        Tuple of (R0_j, T0_j) - pattern-to-line-scan transform.
    """
    R0 = R_linescan_frame @ R_frame_pattern
    T0 = R_linescan_frame @ T_frame_pattern + T_linescan_frame
    
    return R0, T0


def compute_scan_line_in_pattern(
    R0: NDArray[np.float64],
    T0: NDArray[np.float64],
) -> Tuple[float, float, float]:
    """Compute the scan line in pattern coordinates.
    
    The line-scan camera's scan line intersects the pattern plane (Z=0)
    along a line. This line can be computed from the pattern-to-line-scan
    transform.
    
    Implements Equation (30) from the paper. With Z=0, the scan plane
    intersection with the pattern plane gives:
        a*X + b*Y + c = 0
    where a = r0_11, b = r0_12, c = t0_1 (first row of R0 and T0).
    
    Args:
        R0: 3x3 pattern-to-line-scan rotation matrix.
        T0: 3-element pattern-to-line-scan translation.
        
    Returns:
        Line coefficients (a, b, c) where aX + bY + c = 0.
    """
    # The scan plane is the X=0 plane in line-scan coordinates (all rays lie in Y_h-Z_h plane)
    # In pattern coordinates, this becomes:
    # r0_11*X + r0_12*Y + r0_13*Z + t0_1 = 0
    # With Z=0: r0_11*X + r0_12*Y + t0_1 = 0
    
    a = R0[0, 0]  # r0_11
    b = R0[0, 1]  # r0_12
    c = T0[0]     # t0_1
    
    return (a, b, c)
