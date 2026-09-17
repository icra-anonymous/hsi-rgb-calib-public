"""HSI-RGBD Calibration Kit.

A toolkit for calibrating a rigid pushbroom HSI line-scan camera
with an OAK-D S2 RGB-D camera rig.
"""

__version__ = "0.1.0"
__author__ = "HSI-RGBD Calibration Team"

from hsi_rgbd_calib.io.artifact_schema import CalibrationArtifact
from hsi_rgbd_calib.cal_method.interface import CalibrationResult

__all__ = [
    "__version__",
    "CalibrationArtifact",
    "CalibrationResult",
]
