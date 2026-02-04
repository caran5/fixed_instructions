# Scientific Tools Descriptions

## FDA Database

### Overview
The FDA Database tool provides programmatic access to the U.S. Food and Drug Administration's comprehensive databases, including drug approvals, medical device registrations, food safety data, and adverse event reports.

### Key Features
- **Drug Information**: Access to FDA-approved drugs, including NDC codes, labeling, and approval history
- **Medical Devices**: Search device registrations, 510(k) clearances, and recalls
- **Adverse Events**: Query FAERS (FDA Adverse Event Reporting System) data
- **Food Safety**: Access food recalls, outbreak data, and enforcement reports
- **Clinical Trials**: Information on FDA-regulated clinical trials

### Primary Use Cases
1. **Drug Discovery & Development**
   - Research approved drug formulations
   - Analyze drug approval timelines
   - Study drug-drug interactions

2. **Regulatory Compliance**
   - Verify FDA approval status
   - Check product registrations
   - Monitor recalls and enforcement actions

3. **Pharmacovigilance**
   - Analyze adverse event patterns
   - Safety signal detection
   - Post-market surveillance

4. **Market Intelligence**
   - Track new drug approvals
   - Competitive landscape analysis
   - Generic drug availability

### Installation & Setup
**MCP Server Configuration** (Claude Desktop):
```json
{
  "mcpServers": {
    "fda-database": {
      "command": "npx",
      "args": ["-y", "@davila7/claude-code-templates", "fda-database"]
    }
  }
}
```

### API Access Points
- **openFDA API**: drugs, devices, foods, animal-veterinary
- **NDC Directory**: National Drug Code database
- **FAERS**: FDA Adverse Event Reporting System
- **Device Registrations**: Medical device database

### Example Queries
```python
# Search for drug by name
fda.search_drugs(brand_name="Aspirin")

# Get adverse events
fda.get_adverse_events(drug_name="Ibuprofen", limit=100)

# Check device recalls
fda.search_device_recalls(classification="Class I")
```

### Data Format
- JSON responses
- RESTful API endpoints
- Rate limits: 240 requests/minute (1000/hour with API key)

### Resources
- Documentation: https://open.fda.gov/apis/
- API Reference: https://open.fda.gov/apis/
- Status: https://open.fda.gov/status/

---

## TorchDrug

### Overview
TorchDrug is a PyTorch-based machine learning platform for drug discovery that provides comprehensive tools for molecular property prediction, molecule generation, and protein-ligand interaction modeling.

### Key Features
- **Graph Neural Networks**: State-of-the-art GNN architectures for molecular graphs
- **Pre-trained Models**: Access to pre-trained models on large molecular datasets
- **Protein Modeling**: Protein sequence and structure analysis
- **Knowledge Graphs**: Drug-disease-target knowledge graph reasoning
- **Retrosynthesis**: Reaction prediction and retrosynthetic planning

### Core Components

#### 1. **Molecular Representations**
- SMILES strings
- Molecular graphs (atoms as nodes, bonds as edges)
- 3D conformations
- Molecular fingerprints

#### 2. **Model Architectures**
- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks
- **MPNN**: Message Passing Neural Networks
- **SchNet**: Continuous-filter convolutional layers
- **DimeNet**: Directional message passing

#### 3. **Tasks Supported**
- Molecular property prediction (logP, solubility, toxicity)
- Drug-target interaction prediction
- Molecule generation and optimization
- Reaction prediction
- Protein function prediction

### Installation
```bash
# Via pip
pip install torchdrug

# From source
git clone https://github.com/DeepGraphLearning/torchdrug
cd torchdrug
pip install -e .
```

### Dependencies
- PyTorch >= 1.8.0
- RDKit
- NetworkX
- NumPy, SciPy
- CUDA (optional, for GPU acceleration)

### Example Usage
```python
import torch
from torchdrug import core, models, tasks, datasets

# Load dataset
dataset = datasets.ClinTox("~/molecule-datasets/")

# Define model
model = models.GCN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256],
                   batch_norm=True,
                   short_cut=True)

# Create task
task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                 criterion="bce", metric=("auprc", "auroc"))

# Train
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer)
solver.train(num_epoch=100)
```

### Pre-trained Models
- **MolCLR**: Contrastive learning on 10M molecules
- **InfoGraph**: Graph-level representation learning
- **GraphMVP**: Multi-view molecular pre-training
- **GearNet**: Protein structure pre-training

### Benchmark Datasets
- MoleculeNet (BBBP, Tox21, ESOL, FreeSolv, Lipophilicity)
- ZINC (molecule generation)
- PDBbind (protein-ligand binding)
- BindingDB (bioactivity data)

### Resources
- GitHub: https://github.com/DeepGraphLearning/torchdrug
- Documentation: https://torchdrug.ai/docs/
- Paper: https://arxiv.org/abs/2202.08320
- Tutorials: https://torchdrug.ai/docs/tutorials/

---

## DeepChem

### Overview
DeepChem is a comprehensive Python library for deep learning in drug discovery, materials science, quantum chemistry, and biology. It provides high-level APIs for molecular machine learning with support for multiple deep learning frameworks.

### Key Features
- **Multi-Framework Support**: TensorFlow, PyTorch, JAX
- **Rich Featurizers**: 50+ molecular and protein featurization methods
- **Model Zoo**: Pre-built models for various tasks
- **Benchmarking Suite**: MoleculeNet and other standard benchmarks
- **Production-Ready**: Scalable to large datasets

### Core Modules

#### 1. **Data (dc.data)**
- Dataset loading and splitting
- Data transformers (normalization, balancing)
- Molecular file I/O (SDF, CSV, JSON)

#### 2. **Featurizers (dc.feat)**
- **Molecular**:
  - CircularFingerprint (ECFP)
  - MACCSKeysFingerprint
  - RDKitDescriptors
  - MolGraphConvFeaturizer
  - Coulomb matrices
  
- **Protein**:
  - Amino acid composition
  - Secondary structure
  - Binding pocket features

- **Complexes**:
  - Protein-ligand complexes
  - Atomic coordinates

#### 3. **Models (dc.models)**
- **Graph Models**:
  - GraphConvModel
  - MPNNModel (Message Passing)
  - GATModel (Graph Attention)
  - AttentiveFPModel

- **Traditional ML**:
  - Random Forests
  - Gradient Boosting
  - Support Vector Machines

- **Deep Learning**:
  - Multitask networks
  - LSTM for sequences
  - VAE for generation

#### 4. **Metrics (dc.metrics)**
- Classification: ROC-AUC, PRC-AUC, Accuracy
- Regression: MAE, RMSE, R²
- Ranking: Pearson correlation

### Installation (Already Installed)
```bash
# Environment: deepchem_env
conda activate deepchem_env

# Packages installed:
# - deepchem 2.8.0
# - rdkit 2025.9.2
# - tensorflow 2.20.0
# - scikit-learn 1.6.1
```

### Typical Workflow
```python
import deepchem as dc

# 1. Load data
featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(tasks=['activity'],
                           feature_field='smiles',
                           featurizer=featurizer)
dataset = loader.create_dataset('molecules.csv')

# 2. Split data
splitter = dc.splits.RandomSplitter()
train, valid, test = splitter.train_valid_test_split(dataset)

# 3. Build model
model = dc.models.MultitaskClassifier(
    n_tasks=1,
    n_features=1024,
    layer_sizes=[1000, 500]
)

# 4. Train
model.fit(train, nb_epoch=50)

# 5. Evaluate
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
score = model.evaluate(test, [metric])
print(f"Test ROC-AUC: {score['roc_auc_score']}")
```

### MoleculeNet Benchmarks
- **Quantum Mechanics**: QM7, QM8, QM9
- **Physical Chemistry**: ESOL, FreeSolv, Lipophilicity
- **Biophysics**: PCBA, MUV, HIV, BACE
- **Physiology**: BBBP, Tox21, ToxCast, SIDER, ClinTox

### Advanced Features

#### 1. **Transfer Learning**
- Pre-trained ChemBERTa models
- Transfer from large datasets
- Fine-tuning strategies

#### 2. **Active Learning**
- Uncertainty sampling
- Query-by-committee
- Expected improvement

#### 3. **Reinforcement Learning**
- Molecule optimization
- Reward-based generation
- Policy gradient methods

#### 4. **Molecular Generation**
- VAE (Variational Autoencoders)
- GAN (Generative Adversarial Networks)
- Normalizing flows

### Integration with Other Tools
```python
# RDKit integration
from rdkit import Chem
mol = Chem.MolFromSmiles('CCO')
features = dc.feat.RDKitDescriptors().featurize([mol])

# TensorFlow models
model = dc.models.TensorflowModel(...)

# PyTorch models  
model = dc.models.TorchModel(...)
```

### File Formats Supported
- **Input**: CSV, SDF, JSON, PDB, FASTA
- **Output**: HDF5, Pickle, CSV
- **Model Serialization**: SavedModel (TF), PyTorch checkpoints

### Resources
- GitHub: https://github.com/deepchem/deepchem
- Documentation: https://deepchem.readthedocs.io/
- Forum: https://forum.deepchem.io/
- Tutorials: https://github.com/deepchem/deepchem/tree/master/examples
- Paper: https://arxiv.org/abs/1611.03199

### Local Environment
**Location**: `/opt/anaconda3/envs/deepchem_env`
**Example Script**: `/Users/ceejayarana/claude/deepchem_example.py`
**Activation**: `conda activate deepchem_env`

---

## Comparison Matrix

| Feature | FDA Database | TorchDrug | DeepChem |
|---------|-------------|-----------|----------|
| **Primary Focus** | Regulatory data | Drug discovery ML | General cheminformatics ML |
| **Data Source** | FDA official APIs | Research datasets | MoleculeNet + custom |
| **ML Framework** | N/A (API only) | PyTorch only | TensorFlow/PyTorch/JAX |
| **Pre-trained Models** | No | Yes | Yes |
| **Molecular Featurization** | No | Graph-based | 50+ methods |
| **Protein Support** | Limited | Excellent | Good |
| **Production Ready** | Yes | Research-focused | Yes |
| **Learning Curve** | Low | Medium-High | Medium |
| **Best For** | Regulatory research | Graph neural networks | End-to-end pipelines |

---

## Integrated Workflow Example

Combining all three tools for drug discovery:

```python
# 1. Query FDA for approved drugs in a class
import fda_database as fda
approved_drugs = fda.search_drugs(
    indication="hypertension",
    approval_date_range="2020-01-01:2025-12-31"
)

# 2. Use DeepChem to featurize and predict properties
import deepchem as dc
smiles_list = [drug['smiles'] for drug in approved_drugs]
featurizer = dc.feat.CircularFingerprint(size=2048)
features = featurizer.featurize(smiles_list)

# Create dataset
dataset = dc.data.NumpyDataset(X=features, ids=smiles_list)

# Predict toxicity using pre-trained model
tox_model = dc.models.load_from_disk('pretrained_tox21_model')
predictions = tox_model.predict(dataset)

# 3. Use TorchDrug for advanced molecular generation
import torch
from torchdrug import models

# Generate similar molecules with improved properties
generator = models.GCPN(...)
new_molecules = generator.generate(
    seed_smiles=smiles_list[0],
    optimization_target='toxicity',
    num_samples=100
)

# 4. Check if any similar molecules are already FDA approved
for mol in new_molecules:
    existing = fda.search_drugs(smiles=mol.smiles)
    if existing:
        print(f"Found FDA approved: {existing['brand_name']}")
```

---

## Best Practices

### FDA Database
1. Always include API key for production use (higher rate limits)
2. Cache frequently accessed data
3. Handle API rate limits gracefully
4. Validate data freshness (FDA updates regularly)

### TorchDrug
1. Use GPU for training (10-100x speedup)
2. Start with pre-trained models when possible
3. Validate on held-out test sets
4. Monitor for overfitting on small datasets

### DeepChem
1. Choose appropriate featurizers for your task
2. Use data transformers (normalization, balancing)
3. Implement cross-validation for small datasets
4. Save models and datasets for reproducibility
5. Use MoleculeNet for benchmarking

---

*Last Updated: February 4, 2026*
*Environment: deepchem_env (Python 3.9.25)*
