"""Export and import functions for calibration artifacts.

This module provides functions to save and load calibration artifacts
in YAML and JSON formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml

from hsi_rgbd_calib.io.artifact_schema import (
    CalibrationArtifact,
    validate_artifact_dict,
    CameraIntrinsics,
    DepthAlignmentConfig,
    OakCalibration,
    HsiSlitIntrinsics,
    HsiCalibration,
    Extrinsics,
    ExtrinsicUncertainty,
    ValidationSummary,
    FrameDefinition,
)
from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


def export_rig_yaml(
    artifact: CalibrationArtifact,
    output_path: Path | str,
) -> Path:
    """Export a calibration artifact to a YAML file.
    
    Args:
        artifact: The calibration artifact to export.
        output_path: Path to write the YAML file.
        
    Returns:
        Path to the written file.
        
    Raises:
        ValueError: If the artifact is invalid.
        IOError: If the file cannot be written.
    """
    output_path = Path(output_path)
    
    # Validate before export
    is_valid, errors = validate_artifact_dict(artifact.to_dict())
    if not is_valid:
        raise ValueError(f"Invalid artifact: {errors}")
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write YAML with nice formatting
    with open(output_path, "w") as f:
        yaml.dump(
            artifact.to_dict(),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    
    logger.info(f"Exported rig.yaml to {output_path}")
    return output_path


def load_rig_yaml(input_path: Path | str) -> CalibrationArtifact:
    """Load a calibration artifact from a YAML file.
    
    Args:
        input_path: Path to the YAML file.
        
    Returns:
        Loaded CalibrationArtifact.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is invalid.
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Artifact file not found: {input_path}")
    
    with open(input_path, "r") as f:
        data = yaml.safe_load(f)
    
    # Validate
    is_valid, errors = validate_artifact_dict(data)
    if not is_valid:
        raise ValueError(f"Invalid artifact file: {errors}")
    
    return _dict_to_artifact(data)


def _dict_to_artifact(data: Dict[str, Any]) -> CalibrationArtifact:
    """Convert a dictionary to a CalibrationArtifact."""
    # Parse oak section
    oak_data = data["oak"]
    rgb_intr = oak_data["rgb_intrinsics"]
    rgb_intrinsics = CameraIntrinsics(
        fx=rgb_intr["fx"],
        fy=rgb_intr["fy"],
        cx=rgb_intr["cx"],
        cy=rgb_intr["cy"],
        width=rgb_intr["width"],
        height=rgb_intr["height"],
        distortion_model=rgb_intr.get("distortion_model", "radtan"),
        distortion_coeffs=rgb_intr.get("distortion_coeffs", []),
    )
    
    depth_data = oak_data["depth_alignment"]
    depth_alignment = DepthAlignmentConfig(
        aligned_to_rgb=depth_data.get("aligned_to_rgb", True),
        output_width=depth_data.get("output_width"),
        output_height=depth_data.get("output_height"),
        stereo_mode=depth_data.get("stereo_mode", "standard"),
        subpixel_enabled=depth_data.get("subpixel_enabled", True),
        lr_check_enabled=depth_data.get("lr_check_enabled", True),
        notes=depth_data.get("notes", ""),
    )
    
    oak = OakCalibration(
        rgb_intrinsics=rgb_intrinsics,
        depth_alignment=depth_alignment,
        device_serial=oak_data.get("device_serial", ""),
        device_model=oak_data.get("device_model", "OAK-D S2"),
    )
    
    # Parse hsi section
    hsi_data = data["hsi"]
    slit_data = hsi_data["slit_intrinsics"]
    slit_intrinsics = HsiSlitIntrinsics(
        focal_length_slit=slit_data["focal_length_slit"],
        principal_point_u0=slit_data["principal_point_u0"],
        slit_width=slit_data["slit_width"],
        distortion_model=slit_data.get("distortion_model", "none"),
        distortion_coeffs=slit_data.get("distortion_coeffs", []),
    )
    
    hsi = HsiCalibration(
        slit_intrinsics=slit_intrinsics,
        wavelengths_nm=hsi_data.get("wavelengths_nm"),
        wavelengths_file=hsi_data.get("wavelengths_file"),
        num_bands=hsi_data.get("num_bands"),
        integration_time_us=hsi_data.get("integration_time_us"),
    )
    
    # Parse extrinsics
    ext_data = data["extrinsics"]
    uncertainty = None
    if ext_data.get("uncertainty"):
        unc_data = ext_data["uncertainty"]
        uncertainty = ExtrinsicUncertainty(
            translation_std_m=unc_data.get("translation_std_m"),
            rotation_std_deg=unc_data.get("rotation_std_deg"),
        )
    
    extrinsics = Extrinsics(
        T_oakrgb_hsi=ext_data["T_oakrgb_hsi"],
        uncertainty=uncertainty,
        method=ext_data.get("method", "li_wen_qiu"),
        notes=ext_data.get("notes", ""),
    )
    
    # Parse validation summary
    val_data = data.get("validation_summary", {})
    validation_summary = ValidationSummary(
        reprojection_rmse_px=val_data.get("reprojection_rmse_px"),
        median_abs_error_px=val_data.get("median_abs_error_px"),
        max_error_px=val_data.get("max_error_px"),
        repeatability_translation_mm=val_data.get("repeatability_translation_mm"),
        repeatability_rotation_deg=val_data.get("repeatability_rotation_deg"),
        num_validation_points=val_data.get("num_validation_points"),
        slow_motion_check_passed=val_data.get("slow_motion_check_passed"),
        slow_motion_notes=val_data.get("slow_motion_notes", ""),
    )
    
    # Parse frames
    frames = {}
    for name, frame_data in data.get("frames", {}).items():
        frames[name] = FrameDefinition(
            name=frame_data["name"],
            description=frame_data["description"],
            convention=frame_data["convention"],
        )
    
    return CalibrationArtifact(
        artifact_version=data["artifact_version"],
        created_at=data["created_at"],
        frames=frames,
        oak=oak,
        hsi=hsi,
        extrinsics=extrinsics,
        validation_summary=validation_summary,
        session_path=data.get("session_path"),
        notes=data.get("notes", ""),
    )


def export_calibration_report_json(
    artifact: CalibrationArtifact,
    output_path: Path | str,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export a detailed calibration report as JSON.
    
    This creates a more verbose report than rig.yaml, suitable for
    analysis and debugging.
    
    Args:
        artifact: The calibration artifact.
        output_path: Path to write the JSON file.
        additional_info: Optional additional information to include.
        
    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "report_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "artifact": artifact.to_dict(),
    }
    
    if additional_info:
        report["additional_info"] = additional_info
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Exported calibration report to {output_path}")
    return output_path
