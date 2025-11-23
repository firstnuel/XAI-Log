"""
Model Comparison Script

Compares all model results from the results/ directory and generates:
1. Summary table (console + CSV)
2. Bar charts comparing metrics
3. Radar chart for multi-metric comparison
4. F1 improvement chart vs baseline

Usage:
    python scripts/compare_models.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir='results'):
    """Load all JSON result files from directory."""
    results_dir = Path(results_dir)
    results = []

    for json_file in sorted(results_dir.glob('*.json')):
        if json_file.name == 'model_comparison.csv':
            continue  # Skip CSV files
        with open(json_file, 'r') as f:
            data = json.load(f)
            data['_filename'] = json_file.name
            results.append(data)

    return results


def extract_metrics(results):
    """Extract key metrics from results into a DataFrame."""
    rows = []

    for r in results:
        model = r.get('model', 'Unknown')
        dataset = r.get('dataset', 'Unknown')
        metrics = r.get('metrics', {})

        row = {
            'Model': model,
            'Dataset': dataset,
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1': metrics.get('f1', 0),
            'AUC': metrics.get('auc', None),
        }

        # Add training info if available
        if 'training' in r:
            row['Train Time (s)'] = r['training'].get('training_time_seconds', None)
            row['Epochs'] = r['training'].get('num_epochs', None)

        # Add architecture info
        if 'architecture' in r:
            row['Vocab Size'] = r['architecture'].get('vocab_size', None)

        # Add detection info
        if 'detection' in r:
            row['Top-k'] = r['detection'].get('top_k', None)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_comparison_table(df):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - HDFS ANOMALY DETECTION")
    print("=" * 80)

    # Key metrics
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1']
    if 'AUC' in df.columns and df['AUC'].notna().any():
        display_cols.append('AUC')

    print("\n### Performance Metrics ###\n")
    print(df[display_cols].to_string(index=False, float_format='{:.4f}'.format))

    # Find best model for each metric
    print("\n### Best Model per Metric ###\n")
    for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
        best_idx = df[col].idxmax()
        best_model = df.loc[best_idx, 'Model']
        best_value = df.loc[best_idx, col]
        print(f"  {col:12s}: {best_model:20s} ({best_value:.4f})")

    # Calculate improvement over baseline
    baseline_f1 = df['F1'].min()
    best_f1 = df['F1'].max()
    improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100

    print(f"\n### Overall Improvement ###")
    print(f"\n  Baseline F1:     {baseline_f1:.4f}")
    print(f"  Best F1:         {best_f1:.4f}")
    print(f"  Improvement:     +{improvement:.1f}%")

    print("\n" + "=" * 80)


def plot_metric_comparison(df, save_dir='figures'):
    """Create bar charts comparing models."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    models = df['Model'].tolist()

    # Color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df[metric].tolist()
        bars = ax.bar(models, values, color=colors[:len(models)], edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('HDFS Anomaly Detection - Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = save_dir / 'model_comparison_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_radar_chart(df, save_dir='figures'):
    """Create radar chart for multi-metric comparison."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    models = df['Model'].tolist()

    # Number of metrics
    num_metrics = len(metrics)

    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Colors for each model
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, model in enumerate(models):
        values = df[df['Model'] == model][metrics].values.flatten().tolist()
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    save_path = save_dir / 'model_comparison_radar.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_f1_improvement(df, save_dir='figures'):
    """Show F1 improvement from baseline."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sort by F1
    df_sorted = df.sort_values('F1', ascending=True).reset_index(drop=True)

    baseline_f1 = df_sorted['F1'].iloc[0]  # Lowest F1 (likely Isolation Forest)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c' if f1 == baseline_f1 else '#2ecc71' for f1 in df_sorted['F1']]
    bars = ax.barh(df_sorted['Model'], df_sorted['F1'], color=colors, edgecolor='black')

    # Add improvement percentage
    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        f1 = row['F1']
        # Add F1 value
        ax.annotate(f'{f1:.3f}',
                   xy=(f1, bar.get_y() + bar.get_height()/2),
                   xytext=(-35, 0),
                   textcoords="offset points",
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        if f1 != baseline_f1:
            improvement = ((f1 - baseline_f1) / baseline_f1) * 100
            ax.annotate(f'+{improvement:.0f}%',
                       xy=(f1, bar.get_y() + bar.get_height()/2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center', fontsize=11, fontweight='bold', color='green')

    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Comparison (vs Baseline)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

    # Add baseline label
    ax.axvline(x=baseline_f1, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_f1:.3f})')
    ax.legend(loc='lower right')

    save_path = save_dir / 'model_comparison_f1_improvement.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_summary_table(df, save_dir='figures'):
    """Create a visual table as an image."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for table
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1']
    table_data = df[display_cols].copy()

    # Format numbers
    for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
        table_data[col] = table_data[col].apply(lambda x: f'{x:.4f}')

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(display_cols)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color header text white
    for i in range(len(display_cols)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight best F1 row
    best_f1_idx = df['F1'].idxmax() + 1  # +1 for header row
    for j in range(len(display_cols)):
        table[(best_f1_idx, j)].set_facecolor('#C6EFCE')

    plt.title('HDFS Anomaly Detection - Model Comparison', fontsize=14, fontweight='bold', pad=20)

    save_path = save_dir / 'model_comparison_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def save_comparison_csv(df, save_path='results/model_comparison.csv'):
    """Save comparison to CSV."""
    df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")


def main():
    """Main comparison function."""
    print("\n" + "=" * 80)
    print("LOADING MODEL RESULTS")
    print("=" * 80)

    # Load results
    results = load_all_results('results')
    print(f"  Loaded {len(results)} result files")

    if len(results) == 0:
        print("  No result files found in results/ directory!")
        return

    # Extract metrics
    df = extract_metrics(results)

    # Print comparison table
    print_comparison_table(df)

    # Save to CSV
    print("\n### Saving Files ###\n")
    save_comparison_csv(df, 'results/model_comparison.csv')

    # Generate visualizations
    print("\n### Generating Visualizations ###\n")
    plot_metric_comparison(df, 'figures')
    plot_radar_chart(df, 'figures')
    plot_f1_improvement(df, 'figures')
    plot_summary_table(df, 'figures')

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/model_comparison.csv")
    print("  - figures/model_comparison_metrics.png")
    print("  - figures/model_comparison_radar.png")
    print("  - figures/model_comparison_f1_improvement.png")
    print("  - figures/model_comparison_table.png")
    print("")


if __name__ == "__main__":
    main()
