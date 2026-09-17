"""Calibration visualization module.

This module provides visualization functions for Li-Wen-Qiu calibration
results, including pattern plots, residual analysis, 3D rig views, and
diagnostic checks.

Usage:
    from hsi_rgbd_calib.viz import VisualizationData, generate_all_plots
    
    viz_data = VisualizationData.from_calibration_result(result, pattern)
    generate_all_plots(viz_data, output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hsi_rgbd_calib.viz.data_contracts import (
    VisualizationData,
    ViewVizData,
    GroundTruthViz,
)
from hsi_rgbd_calib.viz.transforms import (
    CalibratedRig,
    check_view_validity,
)
from hsi_rgbd_calib.viz.pattern_viz import (
    plot_pattern_with_scanlines,
    get_pattern_bounds,
)
from hsi_rgbd_calib.viz.observation_viz import (
    plot_residuals_bar,
    plot_residuals_histogram,
    plot_init_vs_final,
)
from hsi_rgbd_calib.viz.rig_3d import plot_rig_3d
from hsi_rgbd_calib.viz.diagnostics import (
    plot_chirality_check,
    plot_cost_trace,
    plot_gt_comparison,
)


__all__ = [
    # Data contracts
    "VisualizationData",
    "ViewVizData",
    "GroundTruthViz",
    # Transforms
    "CalibratedRig",
    "check_view_validity",
    # Pattern plots
    "plot_pattern_with_scanlines",
    "get_pattern_bounds",
    # Observation plots
    "plot_residuals_bar",
    "plot_residuals_histogram",
    "plot_init_vs_final",
    # 3D plots
    "plot_rig_3d",
    # Diagnostics
    "plot_chirality_check",
    "plot_cost_trace",
    "plot_gt_comparison",
    # Convenience
    "generate_all_plots",
]


def generate_all_plots(
    viz_data: VisualizationData,
    output_dir: Path,
    include_optional: bool = True,
) -> dict:
    """Generate all visualization plots to a directory.
    
    Creates the standard set of calibration visualization plots.
    
    Args:
        viz_data: Visualization data package.
        output_dir: Directory to save plots (.png files).
        include_optional: Whether to include optional plots (cost_trace, gt).
        
    Returns:
        Dictionary mapping plot name to file path (or None if skipped).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Required plots
    fig = plot_pattern_with_scanlines(viz_data, output_dir / "pattern_scanlines.png")
    results["pattern_scanlines"] = output_dir / "pattern_scanlines.png"
    fig.clf()
    
    fig = plot_residuals_bar(viz_data, output_dir / "residuals_bar.png")
    results["residuals_bar"] = output_dir / "residuals_bar.png"
    fig.clf()
    
    fig = plot_residuals_histogram(viz_data, output_dir / "residuals_hist.png")
    results["residuals_hist"] = output_dir / "residuals_hist.png"
    fig.clf()
    
    fig = plot_rig_3d(viz_data, output_dir / "rig_3d.png")
    results["rig_3d"] = output_dir / "rig_3d.png"
    fig.clf()
    
    fig = plot_init_vs_final(viz_data, output_dir / "init_vs_final.png")
    results["init_vs_final"] = output_dir / "init_vs_final.png"
    fig.clf()
    
    fig = plot_chirality_check(viz_data, output_dir / "chirality_check.png")
    results["chirality_check"] = output_dir / "chirality_check.png"
    fig.clf()
    
    # Optional plots
    if include_optional:
        if viz_data.cost_history:
            fig = plot_cost_trace(viz_data, output_dir / "cost_trace.png")
            if fig:
                results["cost_trace"] = output_dir / "cost_trace.png"
                fig.clf()
            else:
                results["cost_trace"] = None
        else:
            results["cost_trace"] = None
        
        if viz_data.gt:
            fig = plot_gt_comparison(viz_data, output_dir / "gt_comparison.png")
            if fig:
                results["gt_comparison"] = output_dir / "gt_comparison.png"
                fig.clf()
            else:
                results["gt_comparison"] = None
        else:
            results["gt_comparison"] = None
    
    import matplotlib.pyplot as plt
    plt.close('all')
    
    return results
