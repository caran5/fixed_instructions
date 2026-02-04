"""
Visualize Tox21 GCN training results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load results
    with open('tox21_gcn_results.json', 'r') as f:
        results = json.load(f)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: ROC-AUC comparison across splits
    ax1 = axes[0]
    splits = ['Train', 'Valid', 'Test']
    scores = [
        results['scores']['train_roc_auc'],
        results['scores']['valid_roc_auc'],
        results['scores']['test_roc_auc']
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax1.bar(splits, scores, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('ROC-AUC', fontsize=12)
    ax1.set_title('Model Performance by Data Split', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Tasks overview
    ax2 = axes[1]
    tasks = results['tasks']
    y_pos = np.arange(len(tasks))

    # Create horizontal bar chart showing tasks
    ax2.barh(y_pos, [1]*len(tasks), color='#9b59b6', alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tasks, fontsize=10)
    ax2.set_xlabel('Task Included', fontsize=12)
    ax2.set_title(f'Tox21 Tasks (n={len(tasks)})', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.5)
    ax2.set_xticks([])

    # Add task categories annotation
    ax2.text(1.1, 6, 'NR: Nuclear Receptor\nSR: Stress Response',
             fontsize=9, va='center', style='italic', alpha=0.7)

    plt.suptitle(f'GCN on Tox21 Dataset\n{results["epochs"]} Epochs | Scaffold Split | {results["featurizer"]}',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('tox21_gcn_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved visualization to tox21_gcn_results.png")

    # Also show summary stats
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Featurizer: {results['featurizer']}")
    print(f"Splitter: {results['splitter']}")
    print(f"Epochs: {results['epochs']}")
    print(f"Tasks: {len(results['tasks'])}")
    print("-"*50)
    print(f"Train ROC-AUC: {results['scores']['train_roc_auc']:.4f}")
    print(f"Valid ROC-AUC: {results['scores']['valid_roc_auc']:.4f}")
    print(f"Test ROC-AUC:  {results['scores']['test_roc_auc']:.4f}")
    print("="*50)

    plt.show()


if __name__ == '__main__':
    main()
