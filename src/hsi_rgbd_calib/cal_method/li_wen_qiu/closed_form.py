"""Closed-form initialization for Li-Wen-Qiu calibration.

This module implements the paper's DLT-based closed-form solution
using Equations (17)-(26) from Li, Wen, Qiu.

The approach:
1. Build A_J matrix and solve for J via SVD nullspace (Eq. 17)
2. Build A_K matrix and solve for K via SVD nullspace (Eq. 18)
3. Normalize J to get (r11, r12, r13, t1) via Eq. (21)
4. Recover (r22, r23, gamma) from K via Eq. (22)-(25)
5. Recover r3 by cross product Eq. (26)
6. Recover T and (f, v0) from K components
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClosedFormResult:
    """Result from closed-form initialization.
    
    Attributes:
        R: 3x3 rotation matrix (frame-to-line-scan).
        T: 3-element translation vector.
        f: Focal length.
        v0: Principal point.
        success: Whether initialization succeeded.
        message: Status message.
    """
    R: NDArray[np.float64]
    T: NDArray[np.float64]
    f: float
    v0: float
    success: bool
    message: str


def _build_observation_data(
    pattern_points: List[NDArray[np.float64]],
    frame_poses: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
    v_observations: List[NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build flattened X and v arrays from observations.
    
    Returns:
        X_all: (N, 3) array of points in frame coordinates.
        v_all: (N,) array of observed pixel coordinates.
    """
    X_all = []
    v_all = []
    
    for j, (R_j, T_j) in enumerate(frame_poses):
        P_j = pattern_points[j]
        v_j = v_observations[j] if hasattr(v_observations[j], '__len__') else v_observations[j]
        
        for i in range(len(P_j)):
            P_i = np.asarray(P_j[i], dtype=np.float64)
            X_ij = R_j @ P_i + T_j  # Transform to frame coordinates
            X_all.append(X_ij)
            v_all.append(v_j[i])
    
    return np.array(X_all), np.array(v_all)


def _build_A_J(X_all: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build A_J matrix for Eq. (17).
    
    The constraint from the paper is:
    X * r11 + Y * r12 + Z * r13 + t1 = 0
    
    So A_J has rows: [X_ij, Y_ij, Z_ij, 1]
    And J = [r11, r12, r13, t1] is in the nullspace.
    """
    n = len(X_all)
    A = np.zeros((n, 4))
    for i in range(n):
        X, Y, Z = X_all[i]
        A[i] = [X, Y, Z, 1.0]
    return A


def _build_A_K(X_all: NDArray[np.float64], v_all: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build A_K matrix for Eq. (18).
    
    The pixel coordinate equation rearranged to homogeneous form:
    Y*K[0] + Z*K[1] + K[2] - v*Y*K[3] - v*Z*K[4] - v*K[5] = 0
    
    Where K = [f*r33 - v0*r23, -f*r32 + v0*r22, ..., -r23, r22, r11*t3 - r31*t1]
    
    So A_K has rows: [Y_ij, Z_ij, 1, -v*Y_ij, -v*Z_ij, -v]
    """
    n = len(X_all)
    A = np.zeros((n, 6))
    for i in range(n):
        X, Y, Z = X_all[i]
        v = v_all[i]
        A[i] = [Y, Z, 1.0, -v * Y, -v * Z, -v]
    return A


def _recover_parameters_from_JK(
    J: NDArray[np.float64],
    K: NDArray[np.float64],
    X_all: NDArray[np.float64],
    v_all: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float, bool, str]:
    """Recover R, T, f, v0 from J and K using Eq. (21)-(26).
    
    J = [r11, r12, r13, t1] (up to scale)
    K = [f*r33 - v0*r23, -f*r32 + v0*r22, c3, -r23, r22, r11*t3 - r31*t1] (up to scale)
    
    where c3 = f*r11*t2 - f*r21*t1 + v0*r11*t3 - v0*r31*t1
    
    Uses X_all and v_all to validate solutions by reprojection error.
    
    Returns:
        R, T, f, v0, success, message
    """
    # Step 1: Normalize J using Eq. (21)
    # J = alpha * [r11, r12, r13, t1]
    # The first row of R should have unit norm: |[r11, r12, r13]| = 1
    # So alpha = 1 / |J[:3]|
    
    r1_norm = np.linalg.norm(J[:3])
    if r1_norm < 1e-10:
        return np.eye(3), np.zeros(3), 1000.0, 640.0, False, "J[:3] is near zero"
    
    # Two possible signs
    results = []
    
    for sign in [1.0, -1.0]:
        alpha = sign / r1_norm
        
        # Recover first row of R and t1
        r11, r12, r13 = alpha * J[0], alpha * J[1], alpha * J[2]
        t1 = alpha * J[3]
        
        r1 = np.array([r11, r12, r13])
        
        # Step 2: Recover r22, r23 from K using Eq. (22)/(25)
        # K[3] = gamma * (-r23)  => r23 = -K[3] / gamma
        # K[4] = gamma * (r22)   => r22 = K[4] / gamma
        #
        # To find gamma, use the constraint |[r21, r22, r23]| = 1
        # We have: r22^2 + r23^2 <= 1 (equality only if r21 = 0)
        #
        # From K: r22 = K[4]/gamma, r23 = -K[3]/gamma
        # So: r22^2 + r23^2 = (K[4]^2 + K[3]^2) / gamma^2
        #
        # Also, r1 · r2 = 0 (orthogonality):
        # r11*r21 + r12*r22 + r13*r23 = 0
        # r21 = -(r12*r22 + r13*r23) / r11
        #
        # And |r2| = 1:
        # r21^2 + r22^2 + r23^2 = 1
        #
        # Let's denote: a = K[3], b = K[4]
        # r23 = -a/gamma, r22 = b/gamma
        #
        # From orthogonality: r21 = -(r12*b/gamma + r13*(-a/gamma)) / r11
        #                         = -(r12*b - r13*a) / (r11 * gamma)
        # Let c = r12*b - r13*a
        # r21 = -c / (r11 * gamma)
        #
        # From |r2| = 1:
        # c^2 / (r11^2 * gamma^2) + b^2/gamma^2 + a^2/gamma^2 = 1
        # (c^2/r11^2 + a^2 + b^2) / gamma^2 = 1
        # gamma^2 = c^2/r11^2 + a^2 + b^2
        # gamma = sqrt(c^2/r11^2 + a^2 + b^2)
        
        a = K[3]  # corresponds to -r23 * gamma
        b = K[4]  # corresponds to r22 * gamma
        
        if abs(r11) < 1e-10:
            continue  # Skip this sign
        
        c = r12 * b - r13 * a
        
        gamma_sq = c**2 / r11**2 + a**2 + b**2
        if gamma_sq < 1e-20:
            continue
        
        gamma = np.sqrt(gamma_sq)
        
        # Try both signs for gamma
        for gamma_sign in [1.0, -1.0]:
            g = gamma_sign * gamma
            
            r22 = b / g
            r23 = -a / g
            r21 = -c / (r11 * g)
            
            r2 = np.array([r21, r22, r23])
            
            # Check r2 is unit vector
            r2_norm = np.linalg.norm(r2)
            if abs(r2_norm - 1.0) > 0.01:
                continue
            
            # Check orthogonality with r1
            dot_r1_r2 = np.dot(r1, r2)
            if abs(dot_r1_r2) > 0.01:
                continue
            
            # Step 3: Recover r3 by cross product (Eq. 26)
            r3 = np.cross(r1, r2)
            r3 = r3 / np.linalg.norm(r3)  # Ensure unit vector
            
            # Form R
            R = np.array([r1, r2, r3])
            
            # Check R is valid rotation (det = 1)
            det_R = np.linalg.det(R)
            if det_R < 0:
                R[2] = -R[2]  # Flip r3
            
            # Verify R is orthogonal
            RtR = R.T @ R
            if np.linalg.norm(RtR - np.eye(3)) > 0.1:
                continue
            
            r31, r32, r33 = R[2, :]
            
            # Step 4: Recover f and v0 from K[0] and K[1]
            # K[0] = g * (f*r33 - v0*r23)
            # K[1] = g * (-f*r32 + v0*r22)
            #
            # This is a 2x2 linear system:
            # [r33, -r23] [f ]   [K[0]/g]
            # [-r32, r22] [v0] = [K[1]/g]
            
            A_fv0 = np.array([
                [r33, -r23],
                [-r32, r22]
            ])
            b_fv0 = np.array([K[0] / g, K[1] / g])
            
            det_A = r33 * r22 - (-r23) * (-r32)
            if abs(det_A) < 1e-10:
                continue
            
            fv0 = np.linalg.solve(A_fv0, b_fv0)
            f, v0 = fv0[0], fv0[1]
            
            if f < 0:
                # Try flipping sign
                f = -f
                # This might require reconsidering the geometry
            
            if f < 100:  # Unreasonable focal length
                continue
            
            # Step 5: Recover T
            # t1 already known from J
            # K[5] = g * (r11*t3 - r31*t1)  =>  t3 = (K[5]/g + r31*t1) / r11
            
            if abs(r11) < 1e-10:
                continue
            
            t3 = (K[5] / g + r31 * t1) / r11
            
            # K[2] = g * (f*r11*t2 - f*r21*t1 + v0*r11*t3 - v0*r31*t1)
            #      = g * (r11*(f*t2 + v0*t3) - t1*(f*r21 + v0*r31))
            # Let's solve for t2:
            # K[2]/g = f*r11*t2 - f*r21*t1 + v0*r11*t3 - v0*r31*t1
            # f*r11*t2 = K[2]/g + f*r21*t1 - v0*r11*t3 + v0*r31*t1
            # t2 = (K[2]/g + f*r21*t1 - v0*r11*t3 + v0*r31*t1) / (f*r11)
            
            if abs(f * r11) < 1e-10:
                continue
            
            t2 = (K[2] / g + f * r21 * t1 - v0 * r11 * t3 + v0 * r31 * t1) / (f * r11)
            
            T = np.array([t1, t2, t3])
            
            # Chirality check: pattern should be in front of camera
            # For typical setup, we expect t3 > 0 (Z positive)
            if t3 < 0:
                continue  # Pattern behind camera
            
            results.append((R, T, f, v0))
    
    if len(results) == 0:
        return np.eye(3), np.zeros(3), 1000.0, 640.0, False, "No valid solution found"
    
    # Select the best solution by reprojection error
    best_result = None
    best_rmse = float('inf')
    
    for R, T, f, v0 in results:
        # Compute reprojection error
        errors = []
        for X, v_obs in zip(X_all, v_all):
            X_prime = R @ X + T
            if abs(X_prime[2]) < 1e-12:
                continue
            v_pred = f * X_prime[1] / X_prime[2] + v0
            errors.append((v_pred - v_obs) ** 2)
        
        if errors:
            rmse = np.sqrt(np.mean(errors))
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = (R, T, f, v0)
    
    if best_result is None:
        return np.eye(3), np.zeros(3), 1000.0, 640.0, False, "No valid solution found"
    
    R, T, f, v0 = best_result
    return R, T, f, v0, True, f"Closed-form solution found (RMSE={best_rmse:.4f})"


def closed_form_init(
    pattern_points: List[NDArray[np.float64]],
    frame_poses: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
    v_observations: List[NDArray[np.float64]],
) -> ClosedFormResult:
    """Compute initialization for calibration parameters using paper's DLT.
    
    Implements Li-Wen-Qiu Equations (17)-(26).
    
    Args:
        pattern_points: List of n views, each with 6 points in pattern coords.
                        Shape: [(6, 3), ...] * n_views
        frame_poses: List of (R_j, T_j) tuples for each view.
                     R_j: pattern-to-frame rotation (3x3)
                     T_j: pattern-to-frame translation (3,)
        v_observations: List of v1..v6 observations for each view.
                        Shape: [(6,), ...] * n_views
        
    Returns:
        ClosedFormResult with estimated parameters.
    """
    n_views = len(pattern_points)
    
    if n_views < 2:
        return ClosedFormResult(
            R=np.eye(3), T=np.zeros(3), f=1000.0, v0=640.0,
            success=False, message="Need at least 2 views"
        )
    
    # Build observation data
    X_all, v_all = _build_observation_data(pattern_points, frame_poses, v_observations)
    
    if len(X_all) < 6:
        return ClosedFormResult(
            R=np.eye(3), T=np.zeros(3), f=1000.0, v0=640.0,
            success=False, message="Insufficient observations"
        )
    
    logger.debug(f"Built {len(X_all)} observations from {n_views} views")
    
    # Build A_J and solve for J
    A_J = _build_A_J(X_all)
    U_J, S_J, Vh_J = np.linalg.svd(A_J)
    J = Vh_J[-1, :]  # Nullspace vector
    
    logger.debug(f"A_J singular values: {S_J}")
    logger.debug(f"J = {J}")
    
    # Check conditioning: smallest singular value should be near zero
    if S_J[-1] / S_J[0] > 0.01:
        logger.warning(f"A_J poorly conditioned: S[-1]/S[0] = {S_J[-1]/S_J[0]:.6f}")
    
    # Build A_K and solve for K
    A_K = _build_A_K(X_all, v_all)
    U_K, S_K, Vh_K = np.linalg.svd(A_K)
    K = Vh_K[-1, :]  # Nullspace vector
    
    logger.debug(f"A_K singular values: {S_K}")
    logger.debug(f"K = {K}")
    
    # Recover parameters from J and K
    R, T, f, v0, success, message = _recover_parameters_from_JK(J, K, X_all, v_all)
    
    if success:
        logger.info(f"Closed-form: R_err_from_I = {np.linalg.norm(R - np.eye(3)):.4f}, "
                   f"T = {T}, f = {f:.2f}, v0 = {v0:.2f}")
    else:
        logger.warning(f"Closed-form failed: {message}")
    
    return ClosedFormResult(R=R, T=T, f=f, v0=v0, success=success, message=message)
