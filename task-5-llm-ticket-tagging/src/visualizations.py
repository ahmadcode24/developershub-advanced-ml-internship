"""
visualizations.py
-----------------
Publication-quality plotting functions for ticket classification analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_tag_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot the distribution of ticket tags."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tag_counts = df['true_tag'].value_counts().sort_index()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51', '#8B5E3C']

    bars = ax.barh(tag_counts.index, tag_counts.values, color=colors, 
                   edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Number of Tickets', fontsize=12)
    ax.set_title('Support Ticket Distribution by Tag', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, tag_counts.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                str(val), va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '01_tag_distribution.png'), 
                    dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_confidence_comparison(zero_shot_df: pd.DataFrame, few_shot_df: pd.DataFrame,
                                save: bool = True) -> plt.Figure:
    """Plot confidence score distributions for both approaches."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(zero_shot_df['pred_score'], bins=20, color='#2E86AB', 
                 edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Prediction Confidence', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Zero-Shot: Confidence Distribution', fontsize=13, fontweight='bold')
    axes[0].axvline(zero_shot_df['pred_score'].mean(), color='#C73E1D', 
                    linestyle='--', linewidth=2, label=f'Mean: {zero_shot_df["pred_score"].mean():.3f}')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(few_shot_df['pred_score'], bins=20, color='#A23B72', 
                 edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Prediction Confidence', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Few-Shot: Confidence Distribution', fontsize=13, fontweight='bold')
    axes[1].axvline(few_shot_df['pred_score'].mean(), color='#C73E1D', 
                    linestyle='--', linewidth=2, label=f'Mean: {few_shot_df["pred_score"].mean():.3f}')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Model Confidence Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '02_confidence_comparison.png'), 
                    dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_per_tag_accuracy(tag_perf_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot per-tag accuracy comparison between zero-shot and few-shot."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(tag_perf_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, tag_perf_df['Zero-Shot Accuracy'], width,
                   label='Zero-Shot', color='#2E86AB', edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, tag_perf_df['Few-Shot Accuracy'], width,
                   label='Few-Shot', color='#A23B72', edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Ticket Category', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Tag Accuracy: Zero-Shot vs Few-Shot', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tag_perf_df['Tag'], rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '03_per_tag_accuracy.png'), 
                    dpi=200, bbox_inches='tight', facecolor='white')
    return fig
