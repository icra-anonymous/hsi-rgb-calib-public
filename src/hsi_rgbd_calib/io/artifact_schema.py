"""Calibration artifact schema definition.

This module defines the schema for the calibration artifact (rig.yaml),
including dataclasses for all components and validation functions.

The artifact schema follows version 1.0 and includes:
- Artifact metadata (version, timestamp)
- Frame definitions and conventions
- OAK RGB intrinsics and depth alignment configuration
- HSI slit intrinsics and wavelength metadata
- Extrinsic transformation between OAK RGB and HSI
- Validation summary with quality metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from numpy.typing import NDArray


ARTIFACT_VERSION = "1.0"


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters.
    
    Attributes:
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x coordinate (pixels).
        cy: Principal point y coordinate (pixels).
        width: Image width (pixels).
        height: Image height (pixels).
        distortion_model: Distortion model name (e.g., "radtan", "fisheye").
        distortion_coeffs: List of distortion coefficients.
    """
    
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str = "radtan"
    distortion_coeffs: List[float] = field(default_factory=list)
    
    def to_camera_matrix(self) -> NDArray[np.float64]:
        """Convert to 3x3 camera matrix K."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DepthAlignmentConfig:
    """OAK depth alignment configuration.
    
    Attributes:
        aligned_to_rgb: Whether depth is aligned to RGB frame.
        output_width: Aligned depth output width.
        output_height: Aligned depth output height.
        stereo_mode: Stereo depth mode (e.g., "standard", "extended").
        subpixel_enabled: Whether subpixel depth is enabled.
        lr_check_enabled: Whether left-right check is enabled.
        notes: Additional configuration notes.
    """
    
    aligned_to_rgb: bool = True
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    stereo_mode: str = "standard"
    subpixel_enabled: bool = True
    lr_check_enabled: bool = True
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OakCalibration:
    """OAK-D camera calibration data.
    
    Attributes:
        rgb_intrinsics: RGB camera intrinsics.
        depth_alignment: Depth alignment configuration.
        device_serial: OAK device serial number.
        device_model: OAK device model (e.g., "OAK-D S2").
    """
    
    rgb_intrinsics: CameraIntrinsics
    depth_alignment: DepthAlignmentConfig
    device_serial: str = ""
    device_model: str = "OAK-D S2"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rgb_intrinsics": self.rgb_intrinsics.to_dict(),
            "depth_alignment": self.depth_alignment.to_dict(),
            "device_serial": self.device_serial,
            "device_model": self.device_model,
        }


@dataclass
class HsiSlitIntrinsics:
    """HSI line-scan slit intrinsic parameters.
    
    For a pushbroom line-scan camera, we model a 1D intrinsic along the slit:
    
    u = f_slit * tan(theta) + u0
    
    where theta is the angle from the optical axis and u is the pixel coordinate.
    
    Attributes:
        focal_length_slit: Focal length along slit direction (pixels).
        principal_point_u0: Principal point along slit (pixels).
        slit_width: Number of pixels along slit.
        distortion_model: Distortion model (e.g., "none", "polynomial").
        distortion_coeffs: Distortion coefficients if applicable.
    """
    
    focal_length_slit: float
    principal_point_u0: float
    slit_width: int
    distortion_model: str = "none"
    distortion_coeffs: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HsiCalibration:
    """HSI camera calibration data.
    
    Attributes:
        slit_intrinsics: Slit intrinsic parameters.
        wavelengths_nm: List of wavelengths in nanometers (optional).
        wavelengths_file: Path to wavelengths file (optional).
        num_bands: Number of spectral bands.
        integration_time_us: Integration time in microseconds.
    """
    
    slit_intrinsics: HsiSlitIntrinsics
    wavelengths_nm: Optional[List[float]] = None
    wavelengths_file: Optional[str] = None
    num_bands: Optional[int] = None
    integration_time_us: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slit_intrinsics": self.slit_intrinsics.to_dict(),
            "wavelengths_nm": self.wavelengths_nm,
            "wavelengths_file": self.wavelengths_file,
            "num_bands": self.num_bands,
            "integration_time_us": self.integration_time_us,
        }


@dataclass
class ExtrinsicUncertainty:
    """Uncertainty estimates for extrinsic calibration.
    
    Attributes:
        translation_std_m: Standard deviation of translation in meters.
        rotation_std_deg: Standard deviation of rotation in degrees.
    """
    
    translation_std_m: Optional[float] = None
    rotation_std_deg: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Extrinsics:
    """Extrinsic transformation between cameras.
    
    Attributes:
        T_oakrgb_hsi: 4x4 transformation matrix (row-major) from HSI to OAK RGB frame.
        uncertainty: Optional uncertainty estimates.
        method: Method used for calibration.
        notes: Additional notes.
    """
    
    T_oakrgb_hsi: List[List[float]]  # 4x4 row-major
    uncertainty: Optional[ExtrinsicUncertainty] = None
    method: str = "li_wen_qiu"
    notes: str = ""
    
    def to_matrix(self) -> NDArray[np.float64]:
        """Convert to numpy 4x4 matrix."""
        return np.array(self.T_oakrgb_hsi, dtype=np.float64)
    
    @classmethod
    def from_matrix(
        cls, 
        T: NDArray[np.float64], 
        uncertainty: Optional[ExtrinsicUncertainty] = None,
        method: str = "li_wen_qiu",
        notes: str = ""
    ) -> "Extrinsics":
        """Create from numpy matrix."""
        return cls(
            T_oakrgb_hsi=T.tolist(),
            uncertainty=uncertainty,
            method=method,
            notes=notes,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "T_oakrgb_hsi": self.T_oakrgb_hsi,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "method": self.method,
            "notes": self.notes,
        }


@dataclass
class ValidationSummary:
    """Summary of calibration validation metrics.
    
    Attributes:
        reprojection_rmse_px: Root mean square reprojection error in pixels.
        median_abs_error_px: Median absolute reprojection error in pixels.
        max_error_px: Maximum reprojection error in pixels.
        repeatability_translation_mm: Translation repeatability (std) in mm.
        repeatability_rotation_deg: Rotation repeatability (std) in degrees.
        num_validation_points: Number of points used for validation.
        slow_motion_check_passed: Whether slow-motion assumption was validated.
        slow_motion_notes: Notes about slow-motion validation.
    """
    
    reprojection_rmse_px: Optional[float] = None
    median_abs_error_px: Optional[float] = None
    max_error_px: Optional[float] = None
    repeatability_translation_mm: Optional[float] = None
    repeatability_rotation_deg: Optional[float] = None
    num_validation_points: Optional[int] = None
    slow_motion_check_passed: Optional[bool] = None
    slow_motion_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FrameDefinition:
    """Definition of a coordinate frame.
    
    Attributes:
        name: Frame identifier.
        description: Human-readable description.
        convention: Coordinate convention description.
    """
    
    name: str
    description: str
    convention: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CalibrationArtifact:
    """Complete calibration artifact for rig.yaml.
    
    This is the main output of the calibration process and contains all
    information needed for downstream processing.
    
    Attributes:
        artifact_version: Schema version (e.g., "1.0").
        created_at: ISO timestamp of creation.
        frames: Dictionary of frame definitions.
        oak: OAK camera calibration data.
        hsi: HSI camera calibration data.
        extrinsics: Extrinsic transformation.
        validation_summary: Validation metrics.
        session_path: Path to the calibration session (optional).
        notes: Additional notes.
    """
    
    artifact_version: str
    created_at: str
    frames: Dict[str, FrameDefinition]
    oak: OakCalibration
    hsi: HsiCalibration
    extrinsics: Extrinsics
    validation_summary: ValidationSummary
    session_path: Optional[str] = None
    notes: str = ""
    
    @classmethod
    def create(
        cls,
        oak: OakCalibration,
        hsi: HsiCalibration,
        extrinsics: Extrinsics,
        validation_summary: Optional[ValidationSummary] = None,
        session_path: Optional[str] = None,
        notes: str = "",
    ) -> "CalibrationArtifact":
        """Create a new calibration artifact with current timestamp.
        
        Args:
            oak: OAK camera calibration.
            hsi: HSI camera calibration.
            extrinsics: Extrinsic transformation.
            validation_summary: Optional validation metrics.
            session_path: Optional path to session directory.
            notes: Additional notes.
            
        Returns:
            New CalibrationArtifact instance.
        """
        from hsi_rgbd_calib.common.frames import FRAME_CONVENTIONS, Frame
        
        frames = {}
        for frame in [Frame.RIG, Frame.OAK_RGB, Frame.OAK_DEPTH, Frame.HSI]:
            conv = FRAME_CONVENTIONS[frame]
            frames[frame.value] = FrameDefinition(
                name=conv.name,
                description=conv.description,
                convention=f"X: {conv.x_axis}, Y: {conv.y_axis}, Z: {conv.z_axis}",
            )
        
        return cls(
            artifact_version=ARTIFACT_VERSION,
            created_at=datetime.now().isoformat(),
            frames=frames,
            oak=oak,
            hsi=hsi,
            extrinsics=extrinsics,
            validation_summary=validation_summary or ValidationSummary(),
            session_path=session_path,
            notes=notes,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "artifact_version": self.artifact_version,
            "created_at": self.created_at,
            "frames": {k: v.to_dict() for k, v in self.frames.items()},
            "oak": self.oak.to_dict(),
            "hsi": self.hsi.to_dict(),
            "extrinsics": self.extrinsics.to_dict(),
            "validation_summary": self.validation_summary.to_dict(),
            "session_path": self.session_path,
            "notes": self.notes,
        }


def validate_artifact(artifact: CalibrationArtifact) -> Tuple[bool, List[str]]:
    """Validate a CalibrationArtifact instance.
    
    Args:
        artifact: The artifact to validate.
        
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    return validate_artifact_dict(artifact.to_dict())


def validate_artifact_dict(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a calibration artifact dictionary.
    
    Performs structural and type validation on the artifact data.
    
    Args:
        data: Dictionary representation of the artifact.
        
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: List[str] = []
    
    # Required top-level fields
    required_fields = [
        "artifact_version",
        "created_at",
        "frames",
        "oak",
        "hsi",
        "extrinsics",
        "validation_summary",
    ]
    
    for field_name in required_fields:
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")
    
    if errors:
        return False, errors
    
    # Validate artifact_version
    if not isinstance(data["artifact_version"], str):
        errors.append("artifact_version must be a string")
    
    # Validate created_at (should be ISO format)
    if not isinstance(data["created_at"], str):
        errors.append("created_at must be an ISO timestamp string")
    else:
        try:
            datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        except ValueError:
            errors.append("created_at is not a valid ISO timestamp")
    
    # Validate frames
    if not isinstance(data["frames"], dict):
        errors.append("frames must be a dictionary")
    else:
        required_frames = {"rig", "oak_rgb", "oak_depth", "hsi"}
        missing_frames = required_frames - set(data["frames"].keys())
        if missing_frames:
            errors.append(f"Missing frame definitions: {missing_frames}")
    
    # Validate oak
    oak_errors = _validate_oak_section(data.get("oak", {}))
    errors.extend(oak_errors)
    
    # Validate hsi
    hsi_errors = _validate_hsi_section(data.get("hsi", {}))
    errors.extend(hsi_errors)
    
    # Validate extrinsics
    ext_errors = _validate_extrinsics_section(data.get("extrinsics", {}))
    errors.extend(ext_errors)
    
    return len(errors) == 0, errors


def _validate_oak_section(oak: Dict[str, Any]) -> List[str]:
    """Validate the oak section of the artifact."""
    errors = []
    
    if not isinstance(oak, dict):
        return ["oak must be a dictionary"]
    
    if "rgb_intrinsics" not in oak:
        errors.append("oak.rgb_intrinsics is required")
    else:
        intr = oak["rgb_intrinsics"]
        required_intr = ["fx", "fy", "cx", "cy", "width", "height"]
        for field_name in required_intr:
            if field_name not in intr:
                errors.append(f"oak.rgb_intrinsics.{field_name} is required")
    
    if "depth_alignment" not in oak:
        errors.append("oak.depth_alignment is required")
    
    return errors


def _validate_hsi_section(hsi: Dict[str, Any]) -> List[str]:
    """Validate the hsi section of the artifact."""
    errors = []
    
    if not isinstance(hsi, dict):
        return ["hsi must be a dictionary"]
    
    if "slit_intrinsics" not in hsi:
        errors.append("hsi.slit_intrinsics is required")
    else:
        slit = hsi["slit_intrinsics"]
        required_slit = ["focal_length_slit", "principal_point_u0", "slit_width"]
        for field_name in required_slit:
            if field_name not in slit:
                errors.append(f"hsi.slit_intrinsics.{field_name} is required")
    
    return errors


def _validate_extrinsics_section(ext: Dict[str, Any]) -> List[str]:
    """Validate the extrinsics section of the artifact."""
    errors = []
    
    if not isinstance(ext, dict):
        return ["extrinsics must be a dictionary"]
    
    if "T_oakrgb_hsi" not in ext:
        errors.append("extrinsics.T_oakrgb_hsi is required")
    else:
        T = ext["T_oakrgb_hsi"]
        if not isinstance(T, list) or len(T) != 4:
            errors.append("extrinsics.T_oakrgb_hsi must be a 4x4 matrix (list of 4 lists)")
        else:
            for i, row in enumerate(T):
                if not isinstance(row, list) or len(row) != 4:
                    errors.append(f"extrinsics.T_oakrgb_hsi row {i} must have 4 elements")
    
    return errors
