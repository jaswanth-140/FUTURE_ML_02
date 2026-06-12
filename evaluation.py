# file: evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import f1_score, confusion_matrix

PLOTS_DIR = 'plots'


def compute_macro_f1(y_true_col, y_pred_col):
    return f1_score(y_true_col, y_pred_col, average='macro', zero_division=0)


def compute_hamming_loss(y_true, y_pred):
    """Manual Hamming Loss for multiclass-multioutput targets.

    sklearn.metrics.hamming_loss support for the 'multiclass-multioutput'
    target type (two independent multiclass columns) is inconsistent across
    versions, so this computes the definition directly: the mean fraction of
    labels (across both columns) that were predicted incorrectly.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true != y_pred))


def evaluate_predictions(y_true, y_pred, target_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {}
    for i, name in enumerate(target_names):
        results[f'{name} — Macro F1'] = compute_macro_f1(y_true[:, i], y_pred[:, i])

    results['Hamming Loss (overall)'] = compute_hamming_loss(y_true, y_pred)
    return results


def print_metrics(results, label):
    print(f"\n{'-' * 44}")
    print(f"  {label}")
    print(f"{'-' * 44}")
    for key, value in results.items():
        print(f"  {key:<24}: {value:.4f}")
    print()


def print_comparison(results_a, results_b):
    print(f"\n{'=' * 50}")
    print(f"  {'Metric':<24}{'Arch A':>12}{'Arch B':>12}")
    print(f"{'=' * 50}")
    for key in results_a:
        print(f"  {key:<24}{results_a[key]:>12.4f}{results_b[key]:>12.4f}")
    print()


def plot_3d_confusion_matrix(y_true_col, y_pred_col, target_name, arch_name):
    labels = sorted(set(y_true_col) | set(y_pred_col))
    cm = confusion_matrix(y_true_col, y_pred_col, labels=labels)

    n = len(labels)
    x_pos, y_pos = np.meshgrid(np.arange(n), np.arange(n))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(x_pos)
    heights = cm.flatten()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    max_height = max(heights.max(), 1)
    colors = plt.cm.viridis(heights / max_height)

    ax.bar3d(x_pos, y_pos, z_pos, dx=0.8, dy=0.8, dz=heights, color=colors, shade=True)

    ax.set_xticks(np.arange(n) + 0.4)
    ax.set_yticks(np.arange(n) + 0.4)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)

    ax.set_xlabel('Predicted', labelpad=15)
    ax.set_ylabel('Actual', labelpad=15)
    ax.set_zlabel('Count')
    ax.set_title(f'{arch_name} — {target_name} Confusion Volume')

    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    safe_target = target_name.replace(' ', '_')
    filename = os.path.join(PLOTS_DIR, f'confusion_3d_{arch_name}_{safe_target}.png')
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


def generate_all_confusion_plots(y_test, preds_a, preds_b, target_names):
    y_test = np.asarray(y_test)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    for i, name in enumerate(target_names):
        plot_3d_confusion_matrix(y_test[:, i], preds_a[:, i], name, 'Architecture_A')
        plot_3d_confusion_matrix(y_test[:, i], preds_b[:, i], name, 'Architecture_B')