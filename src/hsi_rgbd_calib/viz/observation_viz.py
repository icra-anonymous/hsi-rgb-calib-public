"""Observation visualization.

This module provides functions for visualizing residuals, predictions,
and comparing initial vs final calibration results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from hsi_rgbd_calib.viz.data_contracts import VisualizationData


def plot_residuals_bar(
    viz_data: VisualizationData,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 5),
    title: str = "Per-View Residual RMSE",
) -> plt.Figure:
    """Plot bar chart of per-view residual RMSE.
    
    Args:
        viz_data: Visualization data package.
        output_path: Optional path to save figure.
        figsize: Figure size.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    view_ids = [v.view_id for v in viz_data.views]
    rmses = [v.residual_rmse for v in viz_data.views]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(view_ids)))
    bars = ax.bar(range(len(view_ids)), rmses, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(view_ids)))
    ax.set_xticklabels(view_ids, rotation=45, ha='right')
    ax.set_xlabel('View')
    ax.set_ylabel('RMSE (pixels)')
    ax.set_title(title)
    
    # Add overall RMSE line
    overall_rmse = np.sqrt(np.mean([r**2 for r in rmses]))
    ax.axhline(overall_rmse, color='red', linestyle='--', linewidth=2,
               label=f'Overall RMSE: {overall_rmse:.3f} px')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_residuals_histogram(
    viz_data: VisualizationData,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (8, 5),
    bins: int = 30,
    title: str = "Residual Distribution",
) -> plt.Figure:
    """Plot histogram of all residuals.
    
    Args:
        viz_data: Visualization data package.
        output_path: Optional path to save figure.
        figsize: Figure size.
        bins: Number of histogram bins.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all residuals
    all_residuals = []
    for v in viz_data.views:
        residuals = v.v_final - v.v_observed
        all_residuals.extend(residuals[np.isfinite(residuals)])
    
    all_residuals = np.array(all_residuals)
    
    ax.hist(all_residuals, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add statistics
    mean = np.mean(all_residuals)
    std = np.std(all_residuals)
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
    ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ: {std:.3f}')
    ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Residual (pixels)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_init_vs_final(
    viz_data: VisualizationData,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12, 5),
    title: str = "Initial vs Final Predictions",
) -> plt.Figure:
    """Plot comparison of initial (closed-form) vs final predictions.
    
    Shows how much the nonlinear refinement improved predictions for
    each point and view.
    
    Args:
        viz_data: Visualization data package.
        output_path: Optional path to save figure.
        figsize: Figure size.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Collect errors
    init_errors = []
    final_errors = []
    
    for v in viz_data.views:
        init_err = np.abs(v.v_init - v.v_observed)
        final_err = np.abs(v.v_final - v.v_observed)
        init_errors.extend(init_err[np.isfinite(init_err)])
        final_errors.extend(final_err[np.isfinite(final_err)])
    
    init_errors = np.array(init_errors)
    final_errors = np.array(final_errors)
    
    # Box plot
    ax = axes[0]
    bp = ax.boxplot([init_errors, final_errors], tick_labels=['Initial', 'Final'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel('Absolute Error (pixels)')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # Scatter plot: init vs final
    ax = axes[1]
    ax.scatter(init_errors, final_errors, alpha=0.5, s=20, c='steelblue')
    max_val = max(init_errors.max(), final_errors.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Initial Error (pixels)')
    ax.set_ylabel('Final Error (pixels)')
    ax.set_title('Improvement per Point')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
