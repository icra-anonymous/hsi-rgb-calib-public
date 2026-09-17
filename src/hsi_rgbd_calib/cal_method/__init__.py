"""Calibration method module.

This module provides the interface and backends for the calibration method
(based on the Li-Wen-Qiu line-scan + frame camera calibration approach).
"""

from hsi_rgbd_calib.cal_method.interface import (
    CalibrationResult,
    CalibrationConfig,
    estimate_calibration,
)
from hsi_rgbd_calib.cal_method.stub_backend import (
    StubBackend,
    load_precomputed_results,
)
from hsi_rgbd_calib.cal_method.python_backend import (
    PythonBackend,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu import (
    LiWenQiuBackend,
)

__all__ = [
    "CalibrationResult",
    "CalibrationConfig",
    "estimate_calibration",
    "StubBackend",
    "load_precomputed_results",
    "PythonBackend",
    "LiWenQiuBackend",
]

