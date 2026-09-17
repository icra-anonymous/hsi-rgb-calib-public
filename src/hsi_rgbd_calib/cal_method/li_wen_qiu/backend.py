"""Li-Wen-Qiu calibration backend.

This module provides the main backend class that integrates all stages
of the Li-Wen-Qiu calibration method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from hsi_rgbd_calib.cal_method.interface import (
    CalibrationResult,
    CalibrationConfig,
    HsiSlitIntrinsicsResult,
    ViewResult,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
    recover_pattern_points_from_observations,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.closed_form import (
    closed_form_init,
    ClosedFormResult,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.nonlinear import (
    refine_calibration,
    compute_reprojection_errors,
    RefinementResult,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    compute_transform_pattern_to_linescan,
    compute_scan_line_in_pattern,
    project_to_linescan_batch,
)
from hsi_rgbd_calib.boards.li_wen_qiu_pattern import (
    LiWenQiuPattern,
    load_li_wen_qiu_pattern,
    get_default_li_wen_qiu_pattern,
)
from hsi_rgbd_calib.boards.geometry import intersect_lines_2d
from hsi_rgbd_calib.io.session import SessionData
from hsi_rgbd_calib.common.logging import get_logger
from hsi_rgbd_calib.common.transforms import make_transform

logger = get_logger(__name__)


@dataclass
class ViewObservation:
    """Observation data for a single view.
    
    Attributes:
        R_frame_pattern: 3x3 rotation from pattern to frame camera.
        T_frame_pattern: 3-element translation from pattern to frame.
        v_observations: 6-element array of line-scan pixel coords [v1..v6].
        view_id: Optional identifier for this view.
    """
    R_frame_pattern: NDArray[np.float64]
    T_frame_pattern: NDArray[np.float64]
    v_observations: NDArray[np.float64]
    view_id: Optional[str] = None


class LiWenQiuBackend:
    """Li-Wen-Qiu calibration backend.
    
    This backend implements the complete Li-Wen-Qiu calibration pipeline:
    1. Point correspondence via cross-ratio (Section 3.2)
    2. Closed-form initialization (Section 3.3)
    3. Nonlinear refinement (Section 3.4)
    
    Example:
        >>> backend = LiWenQiuBackend()
        >>> result = backend.estimate(session, config)
    """
    
    def __init__(self, pattern: Optional[LiWenQiuPattern] = None):
        """Initialize backend.
        
        Args:
            pattern: Li-Wen-Qiu pattern model. If None, uses default.
        """
        self.pattern = pattern or get_default_li_wen_qiu_pattern()
    
    def estimate(
        self,
        session: SessionData,
        config: CalibrationConfig,
    ) -> CalibrationResult:
        """Estimate calibration from session data.
        
        This method implements the full Li-Wen-Qiu calibration pipeline.
        
        Args:
            session: Loaded session data containing view observations.
            config: Calibration configuration.
            
        Returns:
            CalibrationResult with estimated parameters.
        """
        logger.info("Running Li-Wen-Qiu calibration backend")
        
        # Load pattern if specified in config
        if config.extra_config.get("pattern_file"):
            pattern_path = Path(config.extra_config["pattern_file"])
            self.pattern = load_li_wen_qiu_pattern(pattern_path)
            logger.info(f"Loaded pattern from {pattern_path}")
        
        # Load view observations from session
        views = self._load_view_observations(session)
        
        if len(views) < 2:
            return CalibrationResult(
                T_oakrgb_hsi=np.eye(4),
                hsi_intrinsics=HsiSlitIntrinsicsResult(
                    focal_length_slit=1000.0,
                    principal_point_u0=640.0,
                    slit_width=1280,
                    distortion_coeffs=[],
                ),
                reprojection_error_rmse=float('inf'),
                reprojection_error_median=float('inf'),
                reprojection_error_max=float('inf'),
                num_correspondences=0,
                num_inliers=0,
                method="li_wen_qiu",
                success=False,
                message=f"Need at least 2 views, got {len(views)}",
            )
        
        logger.info(f"Loaded {len(views)} views for calibration")
        
        # Run calibration pipeline
        return self._run_pipeline(views, config)
    
    def estimate_from_observations(
        self,
        views: List[ViewObservation],
        config: Optional[CalibrationConfig] = None,
    ) -> CalibrationResult:
        """Estimate calibration from pre-loaded view observations.
        
        This is the main entry point for simulation/testing.
        
        Args:
            views: List of ViewObservation objects.
            config: Optional configuration.
            
        Returns:
            CalibrationResult with estimated parameters.
        """
        if config is None:
            config = CalibrationConfig.from_dict({})
        
        return self._run_pipeline(views, config)
    
    def _load_view_observations(self, session: SessionData) -> List[ViewObservation]:
        """Load view observations from session data.
        
        Expected session structure:
            session.cal_method_config["views"] = [
                {
                    "R": [[...], ...],  # 3x3 rotation matrix
                    "T": [...],         # 3-element translation
                    "v": [...],         # 6-element observations
                },
                ...
            ]
        """
        views = []
        
        cal_config = session.cal_method_config or {}
        view_data = cal_config.get("views", [])
        
        for i, v in enumerate(view_data):
            try:
                R = np.array(v["R"], dtype=np.float64)
                T = np.array(v["T"], dtype=np.float64)
                obs = np.array(v["v"], dtype=np.float64)
                
                views.append(ViewObservation(
                    R_frame_pattern=R,
                    T_frame_pattern=T,
                    v_observations=obs,
                    view_id=v.get("id", f"view_{i}"),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid view {i}: {e}")
        
        return views
    
    def _run_pipeline(
        self,
        views: List[ViewObservation],
        config: CalibrationConfig,
    ) -> CalibrationResult:
        """Run the full calibration pipeline.
        
        Steps:
        1. Compute pattern points P_i for each view via cross-ratio
        2. Run closed-form initialization
        3. Run nonlinear refinement
        4. Compute metrics and return result
        """
        # Prepare data structures
        frame_poses = [
            (v.R_frame_pattern, v.T_frame_pattern) for v in views
        ]
        v_observations = [v.v_observations for v in views]
        
        # Step 1: Compute pattern points for each view
        logger.info("Computing pattern points via cross-ratio...")
        pattern_points = []
        
        for j, view in enumerate(views):
            try:
                P_points = recover_pattern_points_from_observations(
                    v_obs=list(view.v_observations),
                    wp1=self.pattern.wp1,
                    wp2=self.pattern.wp2,
                    pattern_lines=self.pattern.feature_lines,
                )
                pattern_points.append(np.array(P_points))
            except ValueError as e:
                logger.warning(f"View {j}: cross-ratio failed - {e}")
                # Use fallback points from pattern geometry
                pattern_points.append(self._get_fallback_points())
        
        # Step 2: Closed-form initialization
        logger.info("Running closed-form initialization...")
        init_result: ClosedFormResult = closed_form_init(
            pattern_points=pattern_points,
            frame_poses=frame_poses,
            v_observations=v_observations,
        )
        
        if not init_result.success:
            logger.warning(f"Closed-form init warning: {init_result.message}")
        
        logger.info(f"Closed-form result: f={init_result.f:.2f}, v0={init_result.v0:.2f}")
        
        # Step 3: Nonlinear refinement
        logger.info("Running nonlinear refinement...")
        refine_result: RefinementResult = refine_calibration(
            R_init=init_result.R,
            T_init=init_result.T,
            f_init=init_result.f,
            v0_init=init_result.v0,
            pattern_lines=self.pattern.feature_lines,
            frame_poses=frame_poses,
            v_observations=v_observations,
            k_init=0.0,
            max_iter=config.max_iterations,
            tol=config.convergence_threshold,
        )
        
        logger.info(
            f"Refinement: f={refine_result.f:.2f}, v0={refine_result.v0:.2f}, "
            f"k={refine_result.k:.2e}"
        )
        
        # Step 4: Compute metrics
        errors = compute_reprojection_errors(
            R=refine_result.R,
            T=refine_result.T,
            f=refine_result.f,
            v0=refine_result.v0,
            k=refine_result.k,
            pattern_lines=self.pattern.feature_lines,
            frame_poses=frame_poses,
            v_observations=v_observations,
        )
        
        valid_errors = errors[np.isfinite(errors)]
        n_total = len(errors)
        n_valid = len(valid_errors)
        
        if n_valid > 0:
            rmse = float(np.sqrt(np.mean(valid_errors ** 2)))
            median = float(np.median(valid_errors))
            max_err = float(np.max(valid_errors))
        else:
            rmse = median = max_err = float('inf')
        
        logger.info(f"Reprojection error: RMSE={rmse:.4f} px, median={median:.4f} px")
        
        # Build 4x4 transform
        T_mat = make_transform(refine_result.R, refine_result.T)
        
        # Step 5: Build per-view results for visualization
        per_view_results = []
        for j, view in enumerate(views):
            R_j, T_j = frame_poses[j]
            v_obs = v_observations[j]
            
            # Compute composed transform
            R0, T0 = compute_transform_pattern_to_linescan(
                R_j, T_j, refine_result.R, refine_result.T
            )
            scan_line = compute_scan_line_in_pattern(R0, T0)
            
            # Initial predictions (using closed-form params)
            R0_init, T0_init = compute_transform_pattern_to_linescan(
                R_j, T_j, init_result.R, init_result.T
            )
            v_init = project_to_linescan_batch(
                pattern_points[j], R0_init, T0_init, 
                init_result.f, init_result.v0, 0.0
            )
            
            # Final predictions
            v_final = project_to_linescan_batch(
                pattern_points[j], R0, T0,
                refine_result.f, refine_result.v0, refine_result.k
            )
            
            # Per-view residuals
            residuals = np.abs(v_final - v_obs)
            valid_res = residuals[np.isfinite(residuals)]
            view_rmse = float(np.sqrt(np.mean(valid_res ** 2))) if len(valid_res) > 0 else float('inf')
            
            # Compute P_pattern_final from scan_line intersections with feature lines
            P_final = []
            for fl in self.pattern.feature_lines:
                pt = intersect_lines_2d(scan_line, fl)
                if pt is not None:
                    P_final.append([pt[0], pt[1], 0.0])
                else:
                    # Parallel - use pattern_points as fallback
                    P_final.append(pattern_points[j][len(P_final)])
            P_pattern_final = np.array(P_final, dtype=np.float64)
            
            per_view_results.append(ViewResult(
                view_id=view.view_id or f"view_{j}",
                R_frame_pattern=R_j,
                T_frame_pattern=T_j,
                R0=R0,
                T0=T0,
                scan_line=scan_line,
                v_observed=v_obs,
                v_init=v_init,
                v_final=v_final,
                P_pattern_init=pattern_points[j],
                P_pattern_final=P_pattern_final,
                residual_rmse=view_rmse,
            ))
        
        # Build result
        intrinsics = HsiSlitIntrinsicsResult(
            focal_length_slit=refine_result.f,
            principal_point_u0=refine_result.v0,
            slit_width=int(refine_result.v0 * 2),  # Approximate
            distortion_coeffs=[refine_result.k] if abs(refine_result.k) > 1e-12 else [],
        )
        
        return CalibrationResult(
            T_oakrgb_hsi=T_mat,
            hsi_intrinsics=intrinsics,
            reprojection_error_rmse=rmse,
            reprojection_error_median=median,
            reprojection_error_max=max_err,
            num_correspondences=n_total,
            num_inliers=n_valid,
            translation_std_m=None,
            rotation_std_deg=None,
            method="li_wen_qiu",
            success=refine_result.success,
            message=refine_result.message,
            per_view=per_view_results,
            cost_history=refine_result.cost_history,
        )
    
    def _get_fallback_points(self) -> NDArray[np.float64]:
        """Get fallback pattern points for degenerate cases."""
        # Use approximate points based on pattern geometry
        wp1 = self.pattern.wp1
        wp2 = self.pattern.wp2
        
        # Approximate P1-P6 assuming vertical scan line at X=wp2
        return np.array([
            [wp2, 0.0, 0.0],       # P1 on L1 (Y=0)
            [wp2, wp2, 0.0],       # P2 on L2 (X=Y)
            [wp2, wp2, 0.0],       # P3 on L3 (Y=wp2)
            [wp2 + wp2, wp2, 0.0], # P4 on L4
            [wp2, wp1, 0.0],       # P5 on L5
            [wp2 + wp1, wp1, 0.0], # P6 on L6
        ], dtype=np.float64)
