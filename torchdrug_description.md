# TorchDrug Tool Description

## Overview
TorchDrug is a PyTorch-based machine learning platform for drug discovery that provides comprehensive tools for molecular property prediction, molecule generation, and protein-ligand interaction modeling. It specializes in graph neural networks and knowledge graph reasoning for pharmaceutical applications.

## Key Features
- **Graph Neural Networks**: State-of-the-art GNN architectures for molecular graphs
- **Pre-trained Models**: Access to pre-trained models on large molecular datasets
- **Protein Modeling**: Protein sequence and structure analysis
- **Knowledge Graphs**: Drug-disease-target knowledge graph reasoning
- **Retrosynthesis**: Reaction prediction and retrosynthetic planning
- **Reinforcement Learning**: Molecule optimization through RL

## Core Components

### 1. Molecular Representations
- **SMILES strings**: Text-based molecular notation
- **Molecular graphs**: Atoms as nodes, bonds as edges
- **3D conformations**: Spatial molecular structures
- **Molecular fingerprints**: Fixed-length binary vectors
- **Quantum descriptors**: Electronic properties

### 2. Model Architectures

#### Graph Neural Networks
- **GCN** (Graph Convolutional Networks): Basic message passing
- **GAT** (Graph Attention Networks): Attention-based aggregation
- **GIN** (Graph Isomorphism Networks): Powerful graph representations
- **MPNN** (Message Passing Neural Networks): Flexible message passing
- **SchNet**: Continuous-filter convolutional layers for 3D molecules
- **DimeNet**: Directional message passing with angle information
- **SphereNet**: Spherical message passing

#### Protein Models
- **GearNet**: Geometry-aware protein graph networks
- **CDConv**: Continuous and discrete convolutions
- **GraphBERT**: Pre-trained protein language models

#### Knowledge Graph Models
- **TransE/TransR**: Translation-based embeddings
- **DistMult**: Bilinear models
- **ComplEx**: Complex embeddings
- **RotatE**: Rotational embeddings

### 3. Tasks Supported

#### Molecular Tasks
- Property prediction (logP, solubility, toxicity, binding affinity)
- Molecule generation and optimization
- Molecular conformation generation
- Reaction prediction and retrosynthesis

#### Protein Tasks
- Protein function prediction
- Protein-protein interaction
- Enzyme classification
- Binding site prediction

#### Knowledge Graph Tasks
- Drug-target interaction prediction
- Drug-disease association
- Protein-protein interaction
- Link prediction and reasoning

## Installation

### Via pip
```bash
pip install torchdrug
```

### From source
```bash
git clone https://github.com/DeepGraphLearning/torchdrug
cd torchdrug
pip install -r requirements.txt
pip install -e .
```

### With Conda
```bash
conda create -n torchdrug python=3.8
conda activate torchdrug
pip install torchdrug
```

## Dependencies

### Required
- **PyTorch** >= 1.8.0 (>= 1.10.0 for protein models)
- **RDKit** >= 2020.03
- **NetworkX** >= 2.5
- **NumPy** >= 1.19
- **SciPy** >= 1.6

### Optional
- **CUDA** >= 10.2 (for GPU acceleration)
- **PyTorch Geometric** (for additional GNN layers)
- **Fair-ESM** (for protein language models)
- **Graphviz** (for visualization)

## Example Usage

### Basic Molecular Property Prediction
```python
import torch
from torchdrug import core, models, tasks, datasets

# Load dataset
dataset = datasets.ClinTox("~/molecule-datasets/")

# Define model
model = models.GCN(
    input_dim=dataset.node_feature_dim,
    hidden_dims=[256, 256, 256],
    batch_norm=True,
    short_cut=True,
    concat_hidden=True
)

# Create task
task = tasks.PropertyPrediction(
    model, 
    task=dataset.tasks,
    criterion="bce", 
    metric=("auprc", "auroc")
)

# Train
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer)
solver.train(num_epoch=100)
solver.evaluate("valid")
```

### Molecule Generation
```python
from torchdrug import models, tasks

# Load pre-trained generator
generator = models.GCPN(
    num_layers=4,
    hidden_dims=128,
    node_hidden_dims=128
)

# Generate molecules
molecules = generator.generate(
    num_sample=100,
    max_resample=5,
    off_policy=False
)

# Optimize for property
optimized = generator.optimize(
    num_sample=100,
    property_name="logP",
    target_value=3.0
)
```

### Protein Function Prediction
```python
from torchdrug import datasets, models, tasks

# Load protein dataset
dataset = datasets.EnzymeCommission("~/protein-datasets/")

# GearNet model for proteins
model = models.GearNet(
    input_dim=21,  # amino acid types
    hidden_dims=[512, 512, 512],
    num_relation=7,  # edge types
    batch_norm=True
)

# Multi-class classification task
task = tasks.MultipleBinaryClassification(
    model,
    task=dataset.tasks,
    criterion="bce",
    metric=("auprc", "auroc")
)

# Train
optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, dataset, None, None, optimizer)
solver.train(num_epoch=50)
```

### Drug-Target Interaction
```python
from torchdrug import models, tasks, datasets

# Load DTI dataset
dataset = datasets.BindingDB("~/dti-datasets/")

# Dual model for drugs and targets
drug_model = models.GIN(input_dim=66, hidden_dims=[256, 256])
protein_model = models.GearNet(input_dim=21, hidden_dims=[512, 512])

# Interaction task
task = tasks.InteractionPrediction(
    drug_model, 
    protein_model,
    task=dataset.tasks
)
```

## Pre-trained Models

### Molecular Pre-training
- **MolCLR**: Contrastive learning on 10M molecules
- **InfoGraph**: Graph-level representation learning
- **GraphMVP**: Multi-view molecular pre-training
- **GROVER**: Graph transformer pre-training

### Protein Pre-training
- **GearNet-Edge**: Pre-trained on AlphaFold structures
- **ESM-1b**: Protein language model (650M parameters)
- **ProteinBERT**: BERT for protein sequences

### Loading Pre-trained Models
```python
from torchdrug import models

# Load pre-trained molecular model
model = models.GIN.load_pretrained("gin_supervised_masking")

# Load pre-trained protein model
protein_model = models.GearNet.load_pretrained("gearnet_edge_alphafold")
```

## Benchmark Datasets

### Molecular Datasets
- **MoleculeNet**: BBBP, Tox21, ToxCast, SIDER, ClinTox, ESOL, FreeSolv, Lipophilicity
- **ZINC**: 250K molecules for generation
- **QM9**: Quantum mechanical properties
- **PDBbind**: Protein-ligand binding affinity

### Protein Datasets
- **EnzymeCommission**: Enzyme classification
- **GeneOntology**: Protein function annotation
- **Fold**: Protein fold classification
- **Reaction**: Enzyme reaction classification

### Knowledge Graphs
- **FB15k-237**: Knowledge base completion
- **WN18RR**: WordNet relations
- **Hetionet**: Biomedical knowledge graph
- **DRKG**: Drug repurposing knowledge graph

## Advanced Features

### Custom Layers
```python
from torchdrug import layers

class CustomConv(layers.MessagePassingBase):
    def message(self, graph, input):
        # Custom message function
        node_in = graph.edge_list[:, 0]
        node_out = graph.edge_list[:, 1]
        message = input[node_in] * input[node_out]
        return message
    
    def aggregate(self, graph, message):
        # Custom aggregation
        return scatter_mean(message, graph.edge_list[:, 1])
```

### Custom Tasks
```python
from torchdrug import tasks

class CustomTask(tasks.Task):
    def __init__(self, model):
        super().__init__(model)
        
    def predict(self, batch):
        # Custom prediction logic
        graph = batch["graph"]
        output = self.model(graph, graph.node_feature)
        return output
    
    def forward(self, batch):
        # Custom forward pass
        pred = self.predict(batch)
        label = batch["label"]
        loss = F.mse_loss(pred, label)
        return {"loss": loss}
```

## GPU Acceleration

### Single GPU
```python
device = torch.device("cuda:0")
model = model.to(device)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0])
```

### Multi-GPU
```python
# Data parallel
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0, 1, 2, 3])

# Distributed training
torch.distributed.init_process_group(backend="nccl")
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0, 1, 2, 3])
```

## Best Practices

### 1. GPU Usage
- Always use GPU for training (10-100x speedup)
- Monitor GPU memory usage
- Use mixed precision training for large models

### 2. Model Selection
- Start with pre-trained models when available
- Use GCN/GIN for molecular graphs
- Use GearNet for protein structures
- Use attention models (GAT) for interpretability

### 3. Data Handling
- Use appropriate data splits (scaffold, random, stratified)
- Normalize features for better convergence
- Augment data with SMILES enumeration

### 4. Training
- Use early stopping to prevent overfitting
- Monitor validation metrics
- Save checkpoints regularly
- Use learning rate scheduling

### 5. Validation
- Always validate on held-out test sets
- Use cross-validation for small datasets
- Report confidence intervals
- Check for data leakage

## Performance Benchmarks

### MoleculeNet (ROC-AUC)
| Dataset | GCN | GIN | GAT | GearNet |
|---------|-----|-----|-----|---------|
| BBBP | 0.91 | 0.93 | 0.92 | N/A |
| Tox21 | 0.83 | 0.85 | 0.84 | N/A |
| SIDER | 0.64 | 0.67 | 0.66 | N/A |

### Protein Tasks (AUPRC)
| Dataset | GearNet | CDConv | ESM-1b |
|---------|---------|--------|--------|
| EC | 0.87 | 0.85 | 0.89 |
| GO-BP | 0.45 | 0.43 | 0.48 |
| Fold | 0.92 | 0.90 | N/A |

## Resources

### Official
- **GitHub**: https://github.com/DeepGraphLearning/torchdrug
- **Documentation**: https://torchdrug.ai/docs/
- **Paper**: https://arxiv.org/abs/2202.08320
- **Tutorials**: https://torchdrug.ai/docs/tutorials/

### Community
- **Discord**: TorchDrug community server
- **Issues**: GitHub issue tracker
- **Forum**: DeepGraphLearning discussions

### Related Projects
- **PyTorch Geometric**: Geometric deep learning
- **DGL-LifeSci**: Life science applications
- **Graphein**: Protein graph construction

## Troubleshooting

### Installation Issues
```bash
# CUDA compatibility
pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# RDKit installation
conda install -c conda-forge rdkit

# Memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Common Errors
- **CUDA out of memory**: Reduce batch size or use gradient accumulation
- **RDKit not found**: Install via conda, not pip
- **Import errors**: Check PyTorch version compatibility

## Citation

```bibtex
@article{zhu2022torchdrug,
  title={TorchDrug: A Powerful and Flexible Machine Learning Platform for Drug Discovery},
  author={Zhu, Zhaocheng and others},
  journal={arXiv preprint arXiv:2202.08320},
  year={2022}
}
```

---

*Last Updated: February 4, 2026*
*Framework: PyTorch-based*
*License: Apache 2.0*
