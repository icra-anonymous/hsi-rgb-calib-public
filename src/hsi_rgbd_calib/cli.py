"""Command-line interface for HSI-RGBD Calibration Kit.

This module provides the main CLI entrypoint with subcommands for:
- calibrate: Run calibration and generate rig.yaml
- validate: Validate calibration with metrics
- export: Export calibration artifact and report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

import yaml

from hsi_rgbd_calib import __version__
from hsi_rgbd_calib.common.logging import (
    setup_logging,
    get_logger,
    print_banner,
    print_success,
    print_error,
    print_info,
    console,
)

logger = get_logger(__name__)


def main() -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    setup_logging(level=log_level)
    
    # Dispatch to subcommand
    if hasattr(args, "func"):
        try:
            return args.func(args)
        except Exception as e:
            print_error(f"Error: {e}")
            if getattr(args, "verbose", False):
                console.print_exception()
            return 1
    else:
        parser.print_help()
        return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hsi-rgbd-calib",
        description="HSI-RGBD Calibration Kit - Calibrate HSI line-scan + OAK-D camera rigs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run calibration
  hsi-rgbd-calib calibrate --session ./my_session --config configs/li_wen_qiu.yaml --out ./output

  # Validate calibration
  hsi-rgbd-calib validate --session ./my_session --rig ./output/rig.yaml --out ./output

  # Export artifacts
  hsi-rgbd-calib export --session ./my_session --rig ./output/rig.yaml --out ./output
""",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
    )
    
    # Calibrate subcommand
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Run calibration and generate rig.yaml",
    )
    calibrate_parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to calibration session directory",
    )
    calibrate_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to calibration configuration file",
    )
    calibrate_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for rig.yaml and report",
    )
    calibrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    calibrate_parser.set_defaults(func=cmd_calibrate)
    
    # Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate calibration with metrics",
    )
    validate_parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to calibration session directory",
    )
    validate_parser.add_argument(
        "--rig",
        type=Path,
        required=True,
        help="Path to rig.yaml calibration artifact",
    )
    validate_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for validation report",
    )
    validate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # Export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Export calibration artifact and report",
    )
    export_parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to calibration session directory",
    )
    export_parser.add_argument(
        "--rig",
        type=Path,
        required=True,
        help="Path to rig.yaml calibration artifact",
    )
    export_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for exported files",
    )
    export_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    export_parser.set_defaults(func=cmd_export)
    
    return parser


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run calibration command."""
    print_banner()
    
    session_path = args.session.resolve()
    config_path = args.config.resolve()
    output_dir = args.out.resolve()
    
    print_info(f"Session: {session_path}")
    print_info(f"Config: {config_path}")
    print_info(f"Output: {output_dir}")
    
    if args.dry_run:
        print_info("[DRY-RUN] Would perform the following:")
        print_info(f"  1. Load session from {session_path}")
        print_info(f"  2. Load config from {config_path}")
        print_info("  3. Run calibration method")
        print_info(f"  4. Export rig.yaml to {output_dir / 'rig.yaml'}")
        print_info(f"  5. Export report to {output_dir / 'calibration_report.json'}")
        return 0
    
    # Import here to avoid circular imports
    from hsi_rgbd_calib.io.session import load_session
    from hsi_rgbd_calib.io.export import export_rig_yaml, export_calibration_report_json
    from hsi_rgbd_calib.io.artifact_schema import (
        CalibrationArtifact,
        OakCalibration,
        HsiCalibration,
        Extrinsics,
        ExtrinsicUncertainty,
        ValidationSummary,
        HsiSlitIntrinsics,
    )
    from hsi_rgbd_calib.cal_method.interface import CalibrationConfig, estimate_calibration
    from hsi_rgbd_calib.oak.intrinsics import load_oak_intrinsics
    from hsi_rgbd_calib.oak.depth_config import load_depth_alignment_config, get_default_depth_config
    
    # Load session
    logger.info("Loading session...")
    session = load_session(session_path)
    
    # Load config
    logger.info("Loading configuration...")
    config = CalibrationConfig.from_yaml(config_path)
    
    # Load OAK intrinsics
    logger.info("Loading OAK intrinsics...")
    oak_intrinsics = load_oak_intrinsics(session_path / "oak" / "intrinsics.yaml")
    
    # Load depth config
    depth_config_path = session_path / "oak" / "depth_config.yaml"
    if depth_config_path.exists():
        depth_config = load_depth_alignment_config(depth_config_path)
        depth_alignment = depth_config.config
    else:
        depth_alignment = get_default_depth_config()
    
    # Run calibration
    logger.info("Running calibration...")
    result = estimate_calibration(session, config)
    
    if not result.success:
        print_error(f"Calibration failed: {result.message}")
        return 1
    
    print_success(f"Calibration completed: RMSE = {result.reprojection_error_rmse:.3f} px")
    
    # Build artifact
    logger.info("Building calibration artifact...")
    
    oak_cal = OakCalibration(
        rgb_intrinsics=oak_intrinsics.rgb,
        depth_alignment=depth_alignment,
        device_serial=oak_intrinsics.device_serial,
        device_model=oak_intrinsics.device_model,
    )
    
    hsi_cal = HsiCalibration(
        slit_intrinsics=HsiSlitIntrinsics(
            focal_length_slit=result.hsi_intrinsics.focal_length_slit,
            principal_point_u0=result.hsi_intrinsics.principal_point_u0,
            slit_width=result.hsi_intrinsics.slit_width,
            distortion_coeffs=result.hsi_intrinsics.distortion_coeffs,
        ),
        wavelengths_nm=session.hsi_metadata.get("wavelengths_nm"),
        num_bands=session.hsi_metadata.get("num_bands"),
    )
    
    uncertainty = None
    if result.translation_std_m is not None or result.rotation_std_deg is not None:
        uncertainty = ExtrinsicUncertainty(
            translation_std_m=result.translation_std_m,
            rotation_std_deg=result.rotation_std_deg,
        )
    
    extrinsics = Extrinsics.from_matrix(
        result.T_oakrgb_hsi,
        uncertainty=uncertainty,
        method=result.method,
    )
    
    validation_summary = ValidationSummary(
        reprojection_rmse_px=result.reprojection_error_rmse,
        median_abs_error_px=result.reprojection_error_median,
        max_error_px=result.reprojection_error_max,
        num_validation_points=result.num_correspondences,
    )
    
    artifact = CalibrationArtifact.create(
        oak=oak_cal,
        hsi=hsi_cal,
        extrinsics=extrinsics,
        validation_summary=validation_summary,
        session_path=str(session_path),
    )
    
    # Export
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rig_path = export_rig_yaml(artifact, output_dir / "rig.yaml")
    print_success(f"Exported: {rig_path}")
    
    report_path = export_calibration_report_json(
        artifact,
        output_dir / "calibration_report.json",
        additional_info={"calibration_result": result.to_dict()},
    )
    print_success(f"Exported: {report_path}")
    
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validation command."""
    print_banner()
    
    session_path = args.session.resolve()
    rig_path = args.rig.resolve()
    output_dir = args.out.resolve()
    
    print_info(f"Session: {session_path}")
    print_info(f"Rig: {rig_path}")
    print_info(f"Output: {output_dir}")
    
    if args.dry_run:
        print_info("[DRY-RUN] Would perform the following:")
        print_info(f"  1. Load session from {session_path}")
        print_info(f"  2. Load rig.yaml from {rig_path}")
        print_info("  3. Compute validation metrics")
        print_info(f"  4. Export validation report to {output_dir / 'validation_report.json'}")
        return 0
    
    import json
    from hsi_rgbd_calib.io.session import load_session
    from hsi_rgbd_calib.io.export import load_rig_yaml
    from hsi_rgbd_calib.metrics.sanity_checks import check_extrinsic_sanity, check_slow_motion_assumption
    
    # Load session and rig
    logger.info("Loading session and calibration...")
    session = load_session(session_path)
    artifact = load_rig_yaml(rig_path)
    
    # Run sanity checks
    logger.info("Running validation checks...")
    
    T = artifact.extrinsics.to_matrix()
    extrinsic_check = check_extrinsic_sanity(T)
    
    # Slow motion check (use defaults if metadata not available)
    hsi_fps = session.hsi_metadata.get("fps", 100.0)
    estimated_velocity = session.hsi_metadata.get("estimated_velocity_mps", 0.1)
    slow_motion_check = check_slow_motion_assumption(hsi_fps, estimated_velocity)
    
    # Build validation report
    report = {
        "session_path": str(session_path),
        "rig_yaml_path": str(rig_path),
        "validation_summary": artifact.validation_summary.to_dict(),
        "extrinsic_sanity_check": extrinsic_check.to_dict(),
        "slow_motion_check": slow_motion_check.to_dict(),
    }
    
    # Export report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_report.json"
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print_success(f"Exported: {report_path}")
    
    # Print summary
    console.print()
    console.print("[bold]Validation Summary:[/bold]")
    console.print(f"  Reprojection RMSE: {artifact.validation_summary.reprojection_rmse_px:.3f} px")
    console.print(f"  Extrinsic check: {'PASS' if extrinsic_check.passed else 'FAIL'}")
    console.print(f"  Slow-motion check: {'PASS' if slow_motion_check.passed else 'FAIL'}")
    
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Run export command."""
    print_banner()
    
    session_path = args.session.resolve()
    rig_path = args.rig.resolve()
    output_dir = args.out.resolve()
    
    print_info(f"Session: {session_path}")
    print_info(f"Rig: {rig_path}")
    print_info(f"Output: {output_dir}")
    
    if args.dry_run:
        print_info("[DRY-RUN] Would perform the following:")
        print_info(f"  1. Load rig.yaml from {rig_path}")
        print_info(f"  2. Copy/export artifacts to {output_dir}")
        return 0
    
    import json
    import shutil
    from hsi_rgbd_calib.io.export import load_rig_yaml, export_rig_yaml, export_calibration_report_json
    
    # Load rig
    logger.info("Loading calibration artifact...")
    artifact = load_rig_yaml(rig_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export rig.yaml (copy or re-export)
    output_rig_path = output_dir / "rig.yaml"
    if rig_path != output_rig_path:
        export_rig_yaml(artifact, output_rig_path)
        print_success(f"Exported: {output_rig_path}")
    
    # Export detailed report
    report_path = export_calibration_report_json(artifact, output_dir / "calibration_report.json")
    print_success(f"Exported: {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
