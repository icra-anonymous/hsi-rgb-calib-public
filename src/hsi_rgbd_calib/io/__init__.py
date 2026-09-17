"""IO utilities for HSI-RGBD calibration."""

from hsi_rgbd_calib.io.artifact_schema import (
    CalibrationArtifact,
    OakCalibration,
    HsiCalibration,
    Extrinsics,
    ValidationSummary,
    validate_artifact,
    validate_artifact_dict,
)
from hsi_rgbd_calib.io.export import (
    export_rig_yaml,
    export_calibration_report_json,
    load_rig_yaml,
)
from hsi_rgbd_calib.io.session import (
    SessionData,
    load_session,
    validate_session_structure,
)

__all__ = [
    # Schema
    "CalibrationArtifact",
    "OakCalibration",
    "HsiCalibration",
    "Extrinsics",
    "ValidationSummary",
    "validate_artifact",
    "validate_artifact_dict",
    # Export
    "export_rig_yaml",
    "export_calibration_report_json",
    "load_rig_yaml",
    # Session
    "SessionData",
    "load_session",
    "validate_session_structure",
]
