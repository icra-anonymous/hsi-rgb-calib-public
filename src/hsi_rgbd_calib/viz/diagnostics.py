"""Diagnostic visualizations.

This module provides diagnostic plots for calibration debugging,
including chirality checks, cost trace, and ground truth comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from hsi_rgbd_calib.viz.data_contracts import VisualizationData
from hsi_rgbd_calib.viz.transforms import check_view_validity


def plot_chirality_check(
    viz_data: VisualizationData,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 5),
    title: str = "Chirality Validation",
) -> plt.Figure:
    """Plot chirality check results for all views.
    
    Shows whether each view passes physical validity checks:
    - Centroid in front of camera
    - Rotation determinant positive
    - At least 4/6 points visible
    
    Args:
        viz_data: Visualization data package.
        output_path: Optional path to save figure.
        figsize: Figure size.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    check_names = ['centroid_in_front', 'det_R0_positive', 'points_visible']
    results = []
    
    for view in viz_data.views:
        checks = check_view_validity(view.R0, view.T0, view.P_pattern_final)
        results.append(checks)
    
    # Create matrix visualization
    n_views = len(viz_data.views)
    n_checks = len(check_names)
    matrix = np.zeros((n_views, n_checks))
    
    for i, res in enumerate(results):
        for j, check in enumerate(check_names):
            matrix[i, j] = 1.0 if res[check] else 0.0
    
    im = ax.imshow(matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(n_views))
    ax.set_xticklabels([v.view_id for v in viz_data.views], rotation=45, ha='right')
    ax.set_yticks(range(n_checks))
    ax.set_yticklabels(['Centroid In Front', 'Det(R0) > 0', '≥4 Points Visible'])
    
    ax.set_xlabel('View')
    ax.set_title(title)
    
    # Add pass/fail annotations
    for i in range(n_views):
        for j in range(n_checks):
            val = matrix[i, j]
            text = '✓' if val > 0.5 else '✗'
            color = 'white' if val < 0.5 else 'black'
            ax.text(i, j, text, ha='center', va='center', fontsize=12, color=color)
    
    # Summary
    n_passed = sum(all(r.values()) for r in results)
    ax.set_xlabel(f'View ({n_passed}/{n_views} all checks passed)')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cost_trace(
    viz_data: VisualizationData,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 5),
    title: str = "Optimization Cost Trace",
) -> Optional[plt.Figure]:
    """Plot optimization cost history.
    
    Shows how the cost function decreased during nonlinear refinement.
    Returns None if cost_history is not available.
    
    Args:
        viz_data: Visualization data package.
        output_path: Optional path to save figure.
        figsize: Figure size.
        title: Plot title.
        
    Returns:
        Matplotlib figure or None if no cost history.
    """
    if viz_data.cost_history is None or len(viz_data.cost_history) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = range(len(viz_data.cost_history))
    costs = viz_data.cost_history
    
    ax.plot(iterations, costs, 'b-', linewidth=2)
    ax.scatter([0], [costs[0]], color='red', s=100, zorder=5, label='Initial')
    ax.scatter([len(costs)-1], [costs[-1]], color='green', s=100, zorder=5, label='Final')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (sum of squared errors)')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add reduction info
    if len(costs) > 1:
        reduction = (costs[0] - costs[-1]) / costs[0] * 100
        ax.annotate(f'Reduction: {reduction:.1f}%',
                   xy=(len(costs)//2, costs[len(costs)//2]),
                   fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_gt_comparison(
    viz_data: VisualizationData,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Ground Truth Comparison",
) -> Optional[plt.Figure]:
    """Plot comparison between estimated and ground truth parameters.
    
    Only available for simulation runs with ground truth data.
    Returns None if no ground truth is available.
    
    Args:
        viz_data: Visualization data package.
        output_path: Optional path to save figure.
        figsize: Figure size.
        title: Plot title.
        
    Returns:
        Matplotlib figure or None if no ground truth.
    """
    if viz_data.gt is None:
        return None
    
    gt = viz_data.gt
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Intrinsics comparison
    ax = axes[0]
    labels = ['f', 'v0', 'k']
    estimated = [viz_data.f, viz_data.v0, viz_data.k * 1e6]  # Scale k for visibility
    ground_truth = [gt.f_true, gt.v0_true, gt.k_true * 1e6]
    
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, estimated, width, label='Estimated', color='steelblue')
    ax.bar(x + width/2, ground_truth, width, label='Ground Truth', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(['f (px)', 'v0 (px)', 'k (×10⁻⁶)'])
    ax.legend()
    ax.set_title('Intrinsics')
    
    # Translation error
    ax = axes[1]
    T_est = viz_data.T_oakrgb_hsi[:3, 3]
    T_gt = gt.T_true
    errors = np.abs(T_est - T_gt) * 1000  # mm
    ax.bar(['X', 'Y', 'Z'], errors, color='steelblue')
    ax.set_ylabel('Error (mm)')
    ax.set_title(f'Translation Error\n(Total: {np.linalg.norm(T_est-T_gt)*1000:.2f} mm)')
    
    # Rotation error  
    ax = axes[2]
    R_est = viz_data.T_oakrgb_hsi[:3, :3]
    R_gt = gt.R_true
    R_err = R_est @ R_gt.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle)
    ax.bar(['Rotation'], [angle_deg], color='coral')
    ax.set_ylabel('Angle Error (deg)')
    ax.set_title('Rotation Error')
    
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
