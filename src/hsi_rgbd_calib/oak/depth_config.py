"""OAK depth alignment configuration handling.

This module handles loading and parsing depth alignment configuration
for OAK-D cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from hsi_rgbd_calib.common.logging import get_logger
from hsi_rgbd_calib.io.artifact_schema import DepthAlignmentConfig

logger = get_logger(__name__)


@dataclass
class DepthConfigData:
    """Container for depth configuration data.
    
    Attributes:
        config: Depth alignment configuration.
        raw_data: Raw configuration data for reference.
    """
    
    config: DepthAlignmentConfig
    raw_data: Dict[str, Any]
    
    def to_artifact_config(self) -> DepthAlignmentConfig:
        """Get the configuration for artifact export."""
        return self.config


def load_depth_alignment_config(path: Path | str) -> DepthConfigData:
    """Load depth alignment configuration from a YAML or JSON file.
    
    Expected file format:
    ```yaml
    depth_alignment:
      aligned_to_rgb: true
      output_width: 1280
      output_height: 720
      
    stereo:
      mode: "standard"
      subpixel_enabled: true
      lr_check_enabled: true
      confidence_threshold: 200
      
    notes: "Configuration used during calibration session"
    ```
    
    Args:
        path: Path to the configuration file.
        
    Returns:
        Loaded DepthConfigData.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Depth config file not found: {path}")
    
    # Load file
    with open(path, "r") as f:
        if path.suffix == ".json":
            import json
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    
    if data is None:
        data = {}
    
    # Parse depth alignment section
    alignment_data = data.get("depth_alignment", {})
    stereo_data = data.get("stereo", {})
    
    config = DepthAlignmentConfig(
        aligned_to_rgb=alignment_data.get("aligned_to_rgb", True),
        output_width=alignment_data.get("output_width"),
        output_height=alignment_data.get("output_height"),
        stereo_mode=stereo_data.get("mode", "standard"),
        subpixel_enabled=stereo_data.get("subpixel_enabled", True),
        lr_check_enabled=stereo_data.get("lr_check_enabled", True),
        notes=data.get("notes", ""),
    )
    
    logger.info(f"Loaded depth config from {path}")
    
    return DepthConfigData(
        config=config,
        raw_data=data,
    )


def get_default_depth_config() -> DepthAlignmentConfig:
    """Get default depth alignment configuration.
    
    Returns:
        Default DepthAlignmentConfig for OAK-D S2.
    """
    return DepthAlignmentConfig(
        aligned_to_rgb=True,
        output_width=1280,
        output_height=720,
        stereo_mode="standard",
        subpixel_enabled=True,
        lr_check_enabled=True,
        notes="Default configuration",
    )
