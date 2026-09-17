"""Session loading and validation.

This module handles loading and validating calibration session directories.

A calibration session has the following structure:
    
    session/
    ├── session.yaml          # Session metadata
    ├── oak/
    │   ├── intrinsics.yaml   # OAK RGB camera intrinsics
    │   ├── depth_config.yaml # Depth alignment configuration
    │   └── images/           # Optional sample images
    ├── hsi/
    │   ├── metadata.yaml     # HSI cube metadata
    │   └── cubes/            # Optional HSI data
    └── cal_method/           # Optional precomputed calibration
        └── cal_method_output.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import yaml

from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SessionData:
    """Container for loaded session data.
    
    Attributes:
        session_path: Path to the session directory.
        session_yaml: Contents of session.yaml.
        oak_intrinsics: OAK intrinsics data.
        oak_depth_config: OAK depth configuration.
        hsi_metadata: HSI metadata.
        cal_method_output: Precomputed calibration output (if available).
    """
    
    session_path: Path
    session_yaml: Dict[str, Any]
    oak_intrinsics: Dict[str, Any]
    oak_depth_config: Dict[str, Any]
    hsi_metadata: Dict[str, Any]
    cal_method_output: Optional[Dict[str, Any]] = None
    
    @property
    def name(self) -> str:
        """Get session name from metadata or directory name."""
        return self.session_yaml.get("name", self.session_path.name)
    
    @property
    def has_precomputed_calibration(self) -> bool:
        """Check if precomputed calibration results are available."""
        return self.cal_method_output is not None


def validate_session_structure(session_dir: Path | str) -> Tuple[bool, List[str]]:
    """Validate that a session directory has the required structure.
    
    Args:
        session_dir: Path to the session directory.
        
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    session_dir = Path(session_dir)
    errors: List[str] = []
    
    if not session_dir.exists():
        return False, [f"Session directory does not exist: {session_dir}"]
    
    if not session_dir.is_dir():
        return False, [f"Session path is not a directory: {session_dir}"]
    
    # Required files/directories
    required = [
        ("session.yaml", "file"),
        ("oak", "dir"),
        ("oak/intrinsics.yaml", "file"),
        ("hsi", "dir"),
        ("hsi/metadata.yaml", "file"),
    ]
    
    for rel_path, item_type in required:
        full_path = session_dir / rel_path
        if not full_path.exists():
            errors.append(f"Missing required {item_type}: {rel_path}")
        elif item_type == "dir" and not full_path.is_dir():
            errors.append(f"Expected directory but found file: {rel_path}")
        elif item_type == "file" and not full_path.is_file():
            errors.append(f"Expected file but found directory: {rel_path}")
    
    return len(errors) == 0, errors


def load_session(session_dir: Path | str) -> SessionData:
    """Load a calibration session from a directory.
    
    Args:
        session_dir: Path to the session directory.
        
    Returns:
        Loaded SessionData.
        
    Raises:
        FileNotFoundError: If the session directory doesn't exist.
        ValueError: If the session structure is invalid.
    """
    session_dir = Path(session_dir)
    
    # Validate structure
    is_valid, errors = validate_session_structure(session_dir)
    if not is_valid:
        raise ValueError(f"Invalid session structure: {errors}")
    
    # Load session.yaml
    with open(session_dir / "session.yaml", "r") as f:
        session_yaml = yaml.safe_load(f) or {}
    
    # Load oak intrinsics
    with open(session_dir / "oak" / "intrinsics.yaml", "r") as f:
        oak_intrinsics = yaml.safe_load(f) or {}
    
    # Load oak depth config (optional)
    depth_config_path = session_dir / "oak" / "depth_config.yaml"
    if depth_config_path.exists():
        with open(depth_config_path, "r") as f:
            oak_depth_config = yaml.safe_load(f) or {}
    else:
        oak_depth_config = {}
        logger.warning("depth_config.yaml not found, using defaults")
    
    # Load hsi metadata
    with open(session_dir / "hsi" / "metadata.yaml", "r") as f:
        hsi_metadata = yaml.safe_load(f) or {}
    
    # Load cal_method output if available
    cal_method_output = None
    cal_method_dir = session_dir / "cal_method"
    if cal_method_dir.exists():
        # Try YAML first, then JSON
        yaml_path = cal_method_dir / "cal_method_output.yaml"
        json_path = cal_method_dir / "cal_method_output.json"
        
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                cal_method_output = yaml.safe_load(f)
            logger.info(f"Loaded precomputed calibration from {yaml_path}")
        elif json_path.exists():
            import json
            with open(json_path, "r") as f:
                cal_method_output = json.load(f)
            logger.info(f"Loaded precomputed calibration from {json_path}")
    
    logger.info(f"Loaded session from {session_dir}")
    
    return SessionData(
        session_path=session_dir,
        session_yaml=session_yaml,
        oak_intrinsics=oak_intrinsics,
        oak_depth_config=oak_depth_config,
        hsi_metadata=hsi_metadata,
        cal_method_output=cal_method_output,
    )
