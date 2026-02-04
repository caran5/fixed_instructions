"""
Train a Graph Convolutional Network (GCN) on the Tox21 dataset.

Loads Tox21 with MolGraphConvFeaturizer and scaffold splitter,
trains GCNModel for 50 epochs, evaluates ROC-AUC, and saves results.
"""

import deepchem as dc
import json
from datetime import datetime


def main():
    print("Loading Tox21 dataset...")

    # Use MolGraphConvFeaturizer for GCNModel (produces DGL-compatible graphs)
    featurizer = dc.feat.MolGraphConvFeaturizer()

    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer=featurizer,
        splitter='scaffold',
        reload=False  # Don't use cached data with different featurizer
    )
    train, valid, test = datasets

    print(f"Tasks: {len(tasks)}")
    print(f"Train samples: {len(train)}")
    print(f"Valid samples: {len(valid)}")
    print(f"Test samples: {len(test)}")

    # Build GCN model (force CPU since DGL doesn't support MPS)
    print("\nBuilding GCN model...")
    import torch
    device = torch.device('cpu')

    model = dc.models.GCNModel(
        n_tasks=len(tasks),
        mode='classification',
        batch_size=128,
        learning_rate=0.001,
        device=device
    )

    # Train for 50 epochs
    print("\nTraining for 50 epochs...")
    losses = []
    for epoch in range(50):
        loss = model.fit(train, nb_epoch=1)
        losses.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50, Loss: {loss:.4f}")

    # Evaluate with ROC-AUC
    print("\nEvaluating model...")
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    train_scores = model.evaluate(train, [metric], transformers)
    valid_scores = model.evaluate(valid, [metric], transformers)
    test_scores = model.evaluate(test, [metric], transformers)

    train_auc = train_scores['roc_auc_score']
    valid_auc = valid_scores['roc_auc_score']
    test_auc = test_scores['roc_auc_score']

    # Print results
    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Valid ROC-AUC: {valid_auc:.4f}")
    print(f"Test ROC-AUC:  {test_auc:.4f}")
    print("=" * 50)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'Tox21',
        'featurizer': 'MolGraphConvFeaturizer',
        'splitter': 'scaffold',
        'model': 'GCNModel',
        'epochs': 50,
        'tasks': tasks,
        'scores': {
            'train_roc_auc': float(train_auc),
            'valid_roc_auc': float(valid_auc),
            'test_roc_auc': float(test_auc)
        }
    }

    output_file = 'tox21_gcn_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return test_auc


if __name__ == '__main__':
    main()
