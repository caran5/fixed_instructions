# DeepChem Tool Description

## Overview
DeepChem is a comprehensive Python library for deep learning in drug discovery, materials science, quantum chemistry, and biology. It provides high-level APIs for molecular machine learning with support for multiple deep learning frameworks (TensorFlow, PyTorch, JAX).

## Key Features
- **Multi-Framework Support**: TensorFlow, PyTorch, JAX backends
- **Rich Featurizers**: 50+ molecular and protein featurization methods
- **Model Zoo**: Pre-built models for various ML tasks
- **Benchmarking Suite**: MoleculeNet and other standard benchmarks
- **Production-Ready**: Scalable to large datasets with parallel processing
- **Active Community**: Regular updates and extensive documentation

## Installation

### Current Environment (Already Installed)
```bash
# Environment: deepchem_env
conda activate deepchem_env

# Installed packages:
# - deepchem 2.8.0
# - rdkit 2025.9.2
# - tensorflow 2.20.0
# - scikit-learn 1.6.1
# - numpy 2.0.2
# - pandas 2.3.3
# - matplotlib 3.9.4
```

### Fresh Installation
```bash
# Via pip (minimal)
pip install deepchem

# With TensorFlow
pip install deepchem[tensorflow]

# With PyTorch
pip install deepchem[torch]

# Full installation
pip install deepchem[tensorflow,torch,jax]
```

### From Source
```bash
git clone https://github.com/deepchem/deepchem.git
cd deepchem
pip install -e .
```

## Core Modules

### 1. Data Management (dc.data)

#### Dataset Classes
```python
import deepchem as dc

# NumpyDataset - in-memory datasets
dataset = dc.data.NumpyDataset(X=features, y=labels, ids=ids)

# DiskDataset - disk-based for large datasets
dataset = dc.data.DiskDataset.from_numpy(X, y, ids)

# ImageDataset - for image data
dataset = dc.data.ImageDataset(X=images, y=labels)
```

#### Data Loaders
```python
# CSV loader
loader = dc.data.CSVLoader(
    tasks=['activity'],
    feature_field='smiles',
    featurizer=dc.feat.CircularFingerprint()
)
dataset = loader.create_dataset('molecules.csv')

# SDF loader
loader = dc.data.SDFLoader(
    tasks=['activity'],
    featurizer=dc.feat.RDKitDescriptors()
)
dataset = loader.create_dataset('molecules.sdf')

# FASTA loader (proteins)
loader = dc.data.FASTALoader()
dataset = loader.create_dataset('proteins.fasta')
```

#### Data Splitters
```python
# Random split
splitter = dc.splits.RandomSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)

# Scaffold split (chemical diversity)
splitter = dc.splits.ScaffoldSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)

# Stratified split (balanced classes)
splitter = dc.splits.SingletaskStratifiedSplitter(task_number=0)
train, test = splitter.train_test_split(dataset)

# Index split (specify indices)
splitter = dc.splits.SpecifiedSplitter(valid_indices=[0,1,2], test_indices=[3,4,5])
train, valid, test = splitter.train_valid_test_split(dataset)
```

### 2. Featurization (dc.feat)

#### Molecular Featurizers

**Fingerprints**
```python
# Circular fingerprints (ECFP)
feat = dc.feat.CircularFingerprint(size=2048, radius=2)

# MACCS keys
feat = dc.feat.MACCSKeysFingerprint()

# RDKit fingerprints
feat = dc.feat.RDKitFingerprint()

# Morgan fingerprints
feat = dc.feat.MorganFingerprint(radius=2, size=1024)
```

**Descriptors**
```python
# RDKit descriptors (200+ molecular properties)
feat = dc.feat.RDKitDescriptors()

# Coulomb matrix (quantum)
feat = dc.feat.CoulombMatrix(max_atoms=50)

# Mol2Vec (learned embeddings)
feat = dc.feat.Mol2VecFingerprint()
```

**Graph Featurizers**
```python
# Graph convolution featurizer
feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)

# Weave featurizer
feat = dc.feat.WeaveFeaturizer()

# MPNN featurizer
feat = dc.feat.PagtnMolGraphFeaturizer()
```

#### Protein Featurizers
```python
# Amino acid composition
feat = dc.feat.RDKitGridFeaturizer()

# One-hot encoding
feat = dc.feat.OneHotFeaturizer(charset='ACDEFGHIKLMNPQRSTVWY')
```

#### Complex Featurizers
```python
# Protein-ligand complexes
feat = dc.feat.RdkitGridFeaturizer(
    box_width=16.0,
    feature_types=['ecfp', 'splif', 'hbond', 'salt_bridge']
)
```

### 3. Models (dc.models)

#### Graph Neural Networks

**Graph Convolutional Model**
```python
model = dc.models.GraphConvModel(
    n_tasks=1,
    graph_conv_layers=[64, 64],
    dense_layer_size=128,
    mode='classification'
)
```

**Message Passing Neural Network**
```python
model = dc.models.MPNNModel(
    n_tasks=1,
    node_out_feats=64,
    edge_hidden_feats=128,
    num_step_message_passing=3
)
```

**Graph Attention Network**
```python
model = dc.models.GATModel(
    n_tasks=1,
    graph_attention_layers=[8, 8],
    n_attention_heads=8
)
```

**Attentive FP**
```python
model = dc.models.AttentiveFPModel(
    n_tasks=1,
    num_layers=2,
    num_timesteps=2,
    graph_feat_size=200
)
```

#### Deep Learning Models

**Multitask Neural Network**
```python
model = dc.models.MultitaskClassifier(
    n_tasks=10,
    n_features=1024,
    layer_sizes=[1000, 500],
    dropout=0.25
)
```

**Convolutional Networks**
```python
# For images/grids
model = dc.models.CNN(
    n_tasks=1,
    n_features=100,
    dims=3
)
```

**LSTM for Sequences**
```python
model = dc.models.TextCNNModel(
    n_tasks=1,
    char_dict=char_dict,
    seq_length=max_length
)
```

#### Traditional ML (Scikit-learn Integration)

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC

# Random Forest
rf = RandomForestRegressor(n_estimators=100)
model = dc.models.SklearnModel(model=rf)

# Gradient Boosting
gb = GradientBoostingClassifier()
model = dc.models.SklearnModel(model=gb)

# SVM
svm = SVC(kernel='rbf')
model = dc.models.SklearnModel(model=svm)
```

#### Generative Models

**Variational Autoencoder**
```python
model = dc.models.SeqToSeq(
    input_tokens=charset,
    output_tokens=charset,
    max_output_length=max_length,
    encoder_layers=2,
    decoder_layers=2
)
```

### 4. Metrics (dc.metrics)

```python
# Classification metrics
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
metric = dc.metrics.Metric(dc.metrics.prc_auc_score)
metric = dc.metrics.Metric(dc.metrics.accuracy_score)
metric = dc.metrics.Metric(dc.metrics.recall_score)

# Regression metrics
metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
metric = dc.metrics.Metric(dc.metrics.r2_score)
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

# Ranking metrics
metric = dc.metrics.Metric(dc.metrics.rms_score)
metric = dc.metrics.Metric(dc.metrics.mae_score)
```

## Complete Workflow Example

```python
import deepchem as dc
import numpy as np

# 1. Load and featurize data
featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(
    tasks=['activity'],
    feature_field='smiles',
    featurizer=featurizer
)
dataset = loader.create_dataset('molecules.csv')

# 2. Split data
splitter = dc.splits.ScaffoldSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)

# 3. Transform/normalize data
transformers = [
    dc.trans.NormalizationTransformer(
        transform_y=True,
        dataset=train
    )
]
for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

# 4. Build model
model = dc.models.MultitaskClassifier(
    n_tasks=1,
    n_features=1024,
    layer_sizes=[1000, 500],
    dropout=0.25
)

# 5. Train
model.fit(train, nb_epoch=50)

# 6. Evaluate
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
train_score = model.evaluate(train, [metric])
valid_score = model.evaluate(valid, [metric])
test_score = model.evaluate(test, [metric])

print(f"Train ROC-AUC: {train_score['roc_auc_score']:.3f}")
print(f"Valid ROC-AUC: {valid_score['roc_auc_score']:.3f}")
print(f"Test ROC-AUC: {test_score['roc_auc_score']:.3f}")

# 7. Make predictions
predictions = model.predict(test)

# 8. Save model
model.save_checkpoint(model_dir='./model_checkpoint')
```

## MoleculeNet Benchmarks

DeepChem includes the MoleculeNet benchmark suite:

### Quantum Mechanics
- **QM7**: 7K molecules, atomization energies
- **QM8**: 22K molecules, electronic properties
- **QM9**: 134K molecules, 12 quantum properties

### Physical Chemistry
- **ESOL**: Aqueous solubility (1K molecules)
- **FreeSolv**: Hydration free energy (642 molecules)
- **Lipophilicity**: Octanol/water partition coefficient (4K molecules)

### Biophysics
- **PCBA**: PubChem bioassays (440K molecules, 128 tasks)
- **MUV**: Maximum unbiased validation (93K molecules, 17 tasks)
- **HIV**: HIV replication inhibition (41K molecules)
- **BACE**: Beta-secretase inhibitors (1.5K molecules)

### Physiology
- **BBBP**: Blood-brain barrier penetration (2K molecules)
- **Tox21**: Toxicity (8K molecules, 12 tasks)
- **ToxCast**: Toxicity pathways (8K molecules, 617 tasks)
- **SIDER**: Side effects (1.4K molecules, 27 tasks)
- **ClinTox**: Clinical trial toxicity (1.5K molecules, 2 tasks)

### Loading Benchmarks
```python
# Load a benchmark dataset
tasks, datasets, transformers = dc.molnet.load_tox21(
    featurizer='ECFP',
    split='scaffold'
)

train, valid, test = datasets
```

## Advanced Features

### 1. Transfer Learning

```python
# Load pre-trained ChemBERTa
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

# Fine-tune on your data
# ... training code ...
```

### 2. Active Learning

```python
# Uncertainty sampling
uncertainties = model.predict_uncertainty(unlabeled_dataset)
selected_indices = np.argsort(uncertainties)[-100:]  # Top 100 uncertain

# Retrain with selected samples
new_train = unlabeled_dataset.select(selected_indices)
model.fit(new_train)
```

### 3. Hyperparameter Optimization

```python
from dc.hyper import HyperparamOpt

# Define parameter search space
params_dict = {
    "layer_sizes": [[1000], [1000, 500], [1000, 500, 250]],
    "dropout": [0.0, 0.25, 0.5],
    "learning_rate": [0.001, 0.0001]
}

# Optimize
optimizer = HyperparamOpt(model_builder)
best_model, best_params, all_results = optimizer.hyperparam_search(
    params_dict,
    train_dataset,
    valid_dataset,
    metric
)
```

### 4. Molecular Generation

```python
# Generate molecules with desired properties
from dc.models import SeqToSeq

# Train generative model
gen_model = SeqToSeq(...)
gen_model.fit(molecular_dataset)

# Generate new molecules
new_molecules = gen_model.predict_on_smiles(["C", "CC", "CCC"])
```

## Integration Examples

### RDKit Integration
```python
from rdkit import Chem
from rdkit.Chem import Descriptors

# Convert SMILES to molecules
mol = Chem.MolFromSmiles('CCO')

# Use RDKit descriptors with DeepChem
feat = dc.feat.RDKitDescriptors()
features = feat.featurize([mol])

# Use with DeepChem dataset
dataset = dc.data.NumpyDataset(X=features, y=labels)
```

### TensorFlow Models
```python
# Build custom TensorFlow model
import tensorflow as tf

class CustomModel(dc.models.KerasModel):
    def __init__(self, n_tasks, n_features):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1000, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_tasks)
        ])
        super().__init__(model, dc.models.losses.L2Loss())
```

### PyTorch Models
```python
# Build custom PyTorch model
import torch
import torch.nn as nn

class CustomPyTorchModel(nn.Module):
    def __init__(self, n_features, n_tasks):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 1000)
        self.fc2 = nn.Linear(1000, n_tasks)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Wrap in DeepChem
model = dc.models.TorchModel(
    CustomPyTorchModel(1024, 1),
    dc.models.losses.L2Loss()
)
```

## Local Environment Setup

### Environment Location
```
/opt/anaconda3/envs/deepchem_env
```

### Activation
```bash
conda activate deepchem_env
```

### Example Script
```bash
# Run the included example
cd /Users/ceejayarana/claude
python deepchem_example.py
```

### Jupyter Notebook
```bash
conda activate deepchem_env
jupyter lab
```

## Best Practices

### 1. Choose Appropriate Featurizers
```python
# Small datasets: Use descriptors or fingerprints
feat = dc.feat.RDKitDescriptors()

# Large datasets: Use learned representations
feat = dc.feat.MolGraphConvFeaturizer()

# 3D structure important: Use Coulomb matrix
feat = dc.feat.CoulombMatrix()
```

### 2. Data Preprocessing
```python
# Normalize features
transformers = [dc.trans.NormalizationTransformer(transform_X=True, dataset=train)]

# Balance datasets
transformers.append(dc.trans.BalancingTransformer(dataset=train))

# Apply transformations
for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
```

### 3. Model Selection
```python
# Small data (<1K): Use traditional ML
model = dc.models.SklearnModel(model=RandomForestClassifier())

# Medium data (1K-100K): Use deep neural networks
model = dc.models.MultitaskClassifier(...)

# Large data (>100K): Use graph neural networks
model = dc.models.GraphConvModel(...)
```

### 4. Cross-Validation
```python
# K-fold cross-validation
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)
scores = []

for train_idx, val_idx in kfold.split(dataset.X):
    train_fold = dataset.select(train_idx)
    val_fold = dataset.select(val_idx)
    
    model.fit(train_fold)
    score = model.evaluate(val_fold, [metric])
    scores.append(score)

print(f"Mean CV score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### 5. Reproducibility
```python
# Set random seeds
import numpy as np
import tensorflow as tf
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Save full configuration
config = {
    'featurizer': 'CircularFingerprint',
    'model': 'MultitaskClassifier',
    'params': model.get_params(),
    'transformers': [str(t) for t in transformers]
}

import json
with open('config.json', 'w') as f:
    json.dump(config, f)
```

## Resources

### Official
- **GitHub**: https://github.com/deepchem/deepchem
- **Documentation**: https://deepchem.readthedocs.io/
- **Forum**: https://forum.deepchem.io/
- **Tutorials**: https://github.com/deepchem/deepchem/tree/master/examples

### Publications
- **Original Paper**: https://arxiv.org/abs/1611.03199
- **MoleculeNet**: https://arxiv.org/abs/1703.00564
- **Applications**: https://deepchem.io/publications/

### Community
- **Discord**: DeepChem community server
- **Twitter**: @deep_chem
- **YouTube**: DeepChem tutorials

## Troubleshooting

### Common Issues

**Issue**: ImportError for TensorFlow/PyTorch
```bash
# Solution: Install missing framework
pip install tensorflow  # or pytorch
```

**Issue**: RDKit molecules not loading
```bash
# Solution: Check SMILES validity
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print("Invalid SMILES")
```

**Issue**: Out of memory
```python
# Solution: Use DiskDataset instead of NumpyDataset
dataset = dc.data.DiskDataset.from_numpy(X, y, ids)

# Or reduce batch size
model.fit(train, batch_size=32)  # default is 50
```

**Issue**: Slow featurization
```python
# Solution: Parallelize
feat = dc.feat.CircularFingerprint()
features = feat.featurize(smiles_list, n_jobs=-1)  # use all cores
```

## Citation

```bibtex
@article{Ramsundar2019,
  title={Deep Learning for the Life Sciences},
  author={Ramsundar, Bharath and Eastman, Peter and Walters, Patrick and Pande, Vijay},
  year={2019},
  publisher={O'Reilly Media}
}
```

---

*Last Updated: February 4, 2026*
*Version: 2.8.0*
*Local Environment: deepchem_env (Python 3.9.25)*
*License: MIT*
