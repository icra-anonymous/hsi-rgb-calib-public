"""OAK camera intrinsics loading.

This module handles loading and parsing OAK-D camera intrinsics
from YAML/JSON metadata files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from numpy.typing import NDArray
import yaml

from hsi_rgbd_calib.common.logging import get_logger
from hsi_rgbd_calib.io.artifact_schema import CameraIntrinsics

logger = get_logger(__name__)


@dataclass
class OakIntrinsicsData:
    """Container for OAK camera intrinsics.
    
    Attributes:
        rgb: RGB camera intrinsics.
        left_mono: Left mono camera intrinsics (optional).
        right_mono: Right mono camera intrinsics (optional).
        device_serial: Device serial number.
        device_model: Device model name.
        calibration_date: Date of camera calibration (from factory or re-calibration).
    """
    
    rgb: CameraIntrinsics
    left_mono: Optional[CameraIntrinsics] = None
    right_mono: Optional[CameraIntrinsics] = None
    device_serial: str = ""
    device_model: str = "OAK-D S2"
    calibration_date: Optional[str] = None
    
    def get_rgb_camera_matrix(self) -> NDArray[np.float64]:
        """Get RGB camera matrix K."""
        return self.rgb.to_camera_matrix()
    
    def get_rgb_distortion_coeffs(self) -> NDArray[np.float64]:
        """Get RGB distortion coefficients."""
        return np.array(self.rgb.distortion_coeffs, dtype=np.float64)


def load_oak_intrinsics(path: Path | str) -> OakIntrinsicsData:
    """Load OAK camera intrinsics from a YAML or JSON file.
    
    Expected file format:
    ```yaml
    device:
      serial: "ABC123"
      model: "OAK-D S2"
    
    rgb:
      fx: 1000.0
      fy: 1000.0
      cx: 640.0
      cy: 360.0
      width: 1280
      height: 720
      distortion_model: "radtan"
      distortion_coeffs: [0.1, -0.2, 0.0, 0.0, 0.05]
    ```
    
    Args:
        path: Path to the intrinsics file.
        
    Returns:
        Loaded OakIntrinsicsData.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    
    # Load file
    with open(path, "r") as f:
        if path.suffix == ".json":
            import json
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    
    if data is None:
        raise ValueError(f"Empty intrinsics file: {path}")
    
    # Parse RGB intrinsics (required)
    rgb_data = data.get("rgb")
    if rgb_data is None:
        raise ValueError("Missing 'rgb' section in intrinsics file")
    
    rgb = _parse_camera_intrinsics(rgb_data, "rgb")
    
    # Parse optional mono camera intrinsics
    left_mono = None
    right_mono = None
    
    if "left_mono" in data:
        left_mono = _parse_camera_intrinsics(data["left_mono"], "left_mono")
    
    if "right_mono" in data:
        right_mono = _parse_camera_intrinsics(data["right_mono"], "right_mono")
    
    # Parse device info
    device_data = data.get("device", {})
    device_serial = device_data.get("serial", "")
    device_model = device_data.get("model", "OAK-D S2")
    calibration_date = device_data.get("calibration_date")
    
    logger.info(f"Loaded OAK intrinsics from {path}")
    
    return OakIntrinsicsData(
        rgb=rgb,
        left_mono=left_mono,
        right_mono=right_mono,
        device_serial=device_serial,
        device_model=device_model,
        calibration_date=calibration_date,
    )


def _parse_camera_intrinsics(data: Dict[str, Any], name: str) -> CameraIntrinsics:
    """Parse camera intrinsics from a dictionary."""
    required_fields = ["fx", "fy", "cx", "cy", "width", "height"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {name} intrinsics")
    
    return CameraIntrinsics(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        width=int(data["width"]),
        height=int(data["height"]),
        distortion_model=data.get("distortion_model", "radtan"),
        distortion_coeffs=data.get("distortion_coeffs", []),
    )
