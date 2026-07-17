"""
visualizations.py
-----------------
Publication-quality plotting functions for ML pipeline analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_target_distribution(y, save=True):
    """Plot churn target distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = pd.Series(y).value_counts().sort_index()
    labels = ['No Churn', 'Churn']
    colors = ['#2E86AB', '#C73E1D']

    bars = axes[0].bar(labels, churn_counts.values, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Churn Distribution', fontweight='bold')
    for bar, val in zip(bars, churn_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(val),
                     ha='center', va='bottom', fontweight='bold')

    axes[1].pie(churn_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Churn Proportion', fontweight='bold')

    plt.suptitle('Customer Churn Target Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'), dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_confusion_matrices(y_test, results_list, save=True):
    """Plot confusion matrices for multiple models."""
    fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]

    for ax, result in zip(axes, results_list):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        ax.set_title(f"{result['model_name']}
Confusion Matrix", fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrix Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '05_confusion_matrices.png'), dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_roc_curves(y_test, results_list, save=True):
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#2E86AB', '#C73E1D']

    for result, color in zip(results_list, colors):
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        ax.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {result['roc_auc']:.3f})",
                color=color, linewidth=2.5)

    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '06_roc_curves.png'), dpi=200, bbox_inches='tight', facecolor='white')
    return fig
