"""Validation metrics for calibration quality assessment."""

from hsi_rgbd_calib.metrics.reprojection import (
    compute_reprojection_error,
    compute_reprojection_metrics,
    project_points_to_camera,
    project_points_to_slit,
)
from hsi_rgbd_calib.metrics.repeatability import (
    compute_repeatability_metrics,
    compare_calibrations,
)
from hsi_rgbd_calib.metrics.sanity_checks import (
    check_slow_motion_assumption,
    check_extrinsic_sanity,
    SlowMotionCheckResult,
)

__all__ = [
    # Reprojection
    "compute_reprojection_error",
    "compute_reprojection_metrics",
    "project_points_to_camera",
    "project_points_to_slit",
    # Repeatability
    "compute_repeatability_metrics",
    "compare_calibrations",
    # Sanity checks
    "check_slow_motion_assumption",
    "check_extrinsic_sanity",
    "SlowMotionCheckResult",
]
