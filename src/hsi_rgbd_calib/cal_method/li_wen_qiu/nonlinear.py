"""Nonlinear refinement for Li-Wen-Qiu calibration.

This module implements Section 3.4 of the Li-Wen-Qiu paper:
nonlinear optimization with intersection point updating.

Reference: Equations (27)-(31) in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.boards.geometry import intersect_lines_2d
from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    project_to_linescan,
    compute_transform_pattern_to_linescan,
    compute_scan_line_in_pattern,
)
from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RefinementResult:
    """Result from nonlinear refinement.
    
    Attributes:
        R: 3x3 rotation matrix (frame-to-line-scan).
        T: 3-element translation vector.
        f: Focal length.
        v0: Principal point.
        k: Distortion coefficient.
        final_cost: Final cost function value.
        initial_cost: Initial cost function value.
        n_iterations: Number of optimizer iterations.
        success: Whether optimization succeeded.
        message: Status message.
        cost_history: List of cost values at each iteration (for visualization).
    """
    R: NDArray[np.float64]
    T: NDArray[np.float64]
    f: float
    v0: float
    k: float
    final_cost: float
    initial_cost: float
    n_iterations: int
    success: bool
    message: str
    cost_history: List[float] = None  # Optional: for cost_trace visualization


def _params_to_RT(params: NDArray) -> Tuple[NDArray, NDArray, float, float, float]:
    """Convert parameter vector to R, T, f, v0, k.
    
    Parameter vector: [f, v0, k, rx, ry, rz, tx, ty, tz]
    where (rx, ry, rz) is the rotation vector (axis-angle).
    """
    f = params[0]
    v0 = params[1]
    k = params[2]
    
    rvec = params[3:6]
    T = params[6:9]
    
    # Convert rotation vector to matrix
    R = Rotation.from_rotvec(rvec).as_matrix()
    
    return R, T, f, v0, k


def _RT_to_params(
    R: NDArray, T: NDArray, f: float, v0: float, k: float
) -> NDArray:
    """Convert R, T, f, v0, k to parameter vector."""
    rvec = Rotation.from_matrix(R).as_rotvec()
    
    return np.array([f, v0, k, rvec[0], rvec[1], rvec[2], T[0], T[1], T[2]])


def compute_cost(
    params: NDArray,
    pattern_lines: List[Tuple[float, float, float]],
    frame_poses: List[Tuple[NDArray, NDArray]],
    v_observations: List[NDArray],
) -> float:
    """Compute the reprojection cost function.
    
    Implements Equation (31) from the paper.
    
    For each view:
    1. Compute pattern-to-line-scan transform using Eq (27)
    2. Compute scan line in pattern plane using Eq (30)
    3. Intersect with feature lines to get updated P_i
    4. Project P_i to predicted v̂_ij using Eq (4)
    5. Accumulate squared residuals
    
    Args:
        params: Parameter vector [f, v0, k, rx, ry, rz, tx, ty, tz].
        pattern_lines: 6 feature lines [(a, b, c), ...].
        frame_poses: List of (R_j, T_j) for each view.
        v_observations: List of [v1..v6] observations for each view.
        
    Returns:
        Sum of squared reprojection errors.
    """
    R, T, f, v0, k = _params_to_RT(params)
    
    total_cost = 0.0
    n_views = len(frame_poses)
    
    for j in range(n_views):
        R_j, T_j = frame_poses[j]
        v_obs = v_observations[j]
        
        # Step 1: Compute pattern-to-line-scan transform (Eq 27)
        R0, T0 = compute_transform_pattern_to_linescan(R_j, T_j, R, T)
        
        # Step 2: Compute scan line in pattern (Eq 30)
        scan_line = compute_scan_line_in_pattern(R0, T0)
        
        # Step 3: Intersect with each feature line to get P_i
        for i in range(6):
            feature_line = pattern_lines[i]
            
            pt = intersect_lines_2d(scan_line, feature_line)
            if pt is None:
                # Parallel lines - add penalty
                total_cost += 1e6
                continue
            
            # P_i in pattern coords (Z=0)
            P_i = np.array([pt[0], pt[1], 0.0])
            
            # Step 4: Project P_i to line-scan using pattern-to-line-scan transform
            v_pred = project_to_linescan(P_i, R0, T0, f, v0, k)
            
            if np.isnan(v_pred):
                total_cost += 1e6
                continue
            
            # Step 5: Accumulate squared error
            residual = v_pred - v_obs[i]
            total_cost += residual ** 2
    
    return total_cost


def refine_calibration(
    R_init: NDArray[np.float64],
    T_init: NDArray[np.float64],
    f_init: float,
    v0_init: float,
    pattern_lines: List[Tuple[float, float, float]],
    frame_poses: List[Tuple[NDArray, NDArray]],
    v_observations: List[NDArray],
    k_init: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> RefinementResult:
    """Refine calibration parameters using nonlinear optimization.
    
    Uses Nelder-Mead simplex method as described in the paper.
    
    Args:
        R_init: Initial 3x3 rotation matrix.
        T_init: Initial 3-element translation.
        f_init: Initial focal length.
        v0_init: Initial principal point.
        pattern_lines: 6 feature lines.
        frame_poses: List of (R_j, T_j) for each view.
        v_observations: List of observations for each view.
        k_init: Initial distortion coefficient (default 0).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        
    Returns:
        RefinementResult with optimized parameters.
    """
    # Convert initial values to parameter vector
    params_init = _RT_to_params(R_init, T_init, f_init, v0_init, k_init)
    
    # Compute initial cost
    initial_cost = compute_cost(
        params_init, pattern_lines, frame_poses, v_observations
    )
    
    logger.info(f"Initial cost: {initial_cost:.6f}")
    
    # Track cost history for visualization
    cost_history = [initial_cost]
    
    def callback(params):
        """Callback to track cost history."""
        cost = compute_cost(params, pattern_lines, frame_poses, v_observations)
        cost_history.append(cost)
    
    # Optimization
    try:
        result = minimize(
            compute_cost,
            params_init,
            args=(pattern_lines, frame_poses, v_observations),
            method="Nelder-Mead",
            callback=callback,
            options={
                "maxiter": max_iter,
                "xatol": tol,
                "fatol": tol,
                "disp": False,
            },
        )
        
        params_opt = result.x
        final_cost = result.fun
        n_iterations = result.nit
        success = result.success
        message = result.message
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return RefinementResult(
            R=R_init, T=T_init, f=f_init, v0=v0_init, k=k_init,
            final_cost=initial_cost, initial_cost=initial_cost,
            n_iterations=0, success=False, message=str(e),
            cost_history=cost_history,
        )
    
    # Extract optimized parameters
    R_opt, T_opt, f_opt, v0_opt, k_opt = _params_to_RT(params_opt)
    
    # Ensure R is a valid rotation matrix
    U, _, Vt = np.linalg.svd(R_opt)
    R_opt = U @ Vt
    if np.linalg.det(R_opt) < 0:
        R_opt[2, :] *= -1
    
    logger.info(f"Final cost: {final_cost:.6f} (reduction: {(initial_cost - final_cost) / initial_cost * 100:.2f}%)")
    
    return RefinementResult(
        R=R_opt,
        T=T_opt,
        f=f_opt,
        v0=v0_opt,
        k=k_opt,
        final_cost=final_cost,
        initial_cost=initial_cost,
        n_iterations=n_iterations,
        success=success,
        message=message,
        cost_history=cost_history,
    )


def compute_reprojection_errors(
    R: NDArray[np.float64],
    T: NDArray[np.float64],
    f: float,
    v0: float,
    k: float,
    pattern_lines: List[Tuple[float, float, float]],
    frame_poses: List[Tuple[NDArray, NDArray]],
    v_observations: List[NDArray],
) -> NDArray[np.float64]:
    """Compute individual reprojection errors for all points.
    
    Args:
        R, T, f, v0, k: Calibration parameters.
        pattern_lines: Feature lines.
        frame_poses: View poses.
        v_observations: Observed coordinates.
        
    Returns:
        Array of reprojection errors for each point.
    """
    errors = []
    
    for j, (R_j, T_j) in enumerate(frame_poses):
        v_obs = v_observations[j]
        
        R0, T0 = compute_transform_pattern_to_linescan(R_j, T_j, R, T)
        scan_line = compute_scan_line_in_pattern(R0, T0)
        
        for i in range(6):
            pt = intersect_lines_2d(scan_line, pattern_lines[i])
            if pt is None:
                errors.append(float('inf'))
                continue
            
            P_i = np.array([pt[0], pt[1], 0.0])
            v_pred = project_to_linescan(P_i, R0, T0, f, v0, k)
            
            if np.isnan(v_pred):
                errors.append(float('inf'))
            else:
                errors.append(abs(v_pred - v_obs[i]))
    
    return np.array(errors)
