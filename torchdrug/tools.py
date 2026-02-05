"""
TorchDrug Tool Descriptions for Function Calling

Comprehensive tool definitions for drug discovery with PyTorch.
Covers molecular property prediction, protein representation, 
knowledge graphs, and generative models.
"""

# =============================================================================
# DATA LOADING (3)
# =============================================================================

torchdrug_load_molecule_dataset = {
    "name": "torchdrug_load_molecule_dataset",
    "description": """Load molecular datasets from built-in benchmarks or custom files.
        Supports MoleculeNet, ZINC, ChEMBL, and custom CSV/SDF formats.
        Automatic SMILES parsing and graph construction.""",
    
    "required_parameters": [
        {"name": "dataset_name", "type": "string", "description": "Dataset name (BBBP, Tox21, ESOL, etc.) or path to file"}
    ],
    
    "optional_parameters": [
        {"name": "transform", "type": "object", "default": None, "description": "Transform to apply to molecules"},
        {"name": "lazy", "type": "boolean", "default": False, "description": "Lazy loading for large datasets"},
        {"name": "verbose", "type": "integer", "default": 1, "description": "Verbosity level"}
    ],
    
    "hardware_requirements": ["CPU required", "GPU optional", "RAM: 4-16GB depending on dataset"],
    "time_complexity": "O(n) - Linear with dataset size",
    
    "outputs": [
        {"name": "dataset", "type": "torchdrug.data.MoleculeDataset", "description": "TorchDrug dataset object"},
        {"name": "num_molecules", "type": "integer", "description": "Number of molecules loaded"},
        {"name": "tasks", "type": "list", "description": "List of prediction task names"}
    ],
    
    "failure_modes": [
        {"condition": "Unknown dataset name", "behavior": "Raises ValueError"},
        {"condition": "Invalid file format", "behavior": "Raises IOError"},
        {"condition": "Memory overflow", "behavior": "Use lazy=True"}
    ]
}

torchdrug_load_protein_dataset = {
    "name": "torchdrug_load_protein_dataset",
    "description": """Load protein datasets for structure and function prediction.
        Supports AlphaFoldDB, PDB, and sequence-based datasets.
        Automatic structure parsing and residue graph construction.""",
    
    "required_parameters": [
        {"name": "dataset_name", "type": "string", "description": "Dataset name (EnzymeCommission, GeneOntology, etc.)"}
    ],
    
    "optional_parameters": [
        {"name": "transform", "type": "object", "default": None, "description": "Transform to apply to proteins"},
        {"name": "atom_feature", "type": "string", "default": "default", "description": "Atom feature type"},
        {"name": "bond_feature", "type": "string", "default": "default", "description": "Bond feature type"},
        {"name": "residue_feature", "type": "string", "default": "default", "description": "Residue feature type"}
    ],
    
    "hardware_requirements": ["CPU required", "GPU optional", "RAM: 8-32GB"],
    "time_complexity": "O(n) - Linear with dataset size",
    
    "outputs": [
        {"name": "dataset", "type": "torchdrug.data.ProteinDataset", "description": "TorchDrug protein dataset"},
        {"name": "num_proteins", "type": "integer", "description": "Number of proteins loaded"},
        {"name": "tasks", "type": "list", "description": "List of prediction task names"}
    ],
    
    "failure_modes": [
        {"condition": "Unknown dataset name", "behavior": "Raises ValueError"},
        {"condition": "PDB parsing error", "behavior": "Logs warning, skips structure"}
    ]
}

torchdrug_load_knowledge_graph = {
    "name": "torchdrug_load_knowledge_graph",
    "description": """Load biomedical knowledge graph datasets.
        Supports DrugBank, Hetionet, and custom triples.
        Automatic entity and relation indexing.""",
    
    "required_parameters": [
        {"name": "dataset_name", "type": "string", "description": "Dataset name (DrugBank, Hetionet, etc.)"}
    ],
    
    "optional_parameters": [
        {"name": "include_relations", "type": "array", "default": None, "description": "Specific relation types to include"},
        {"name": "exclude_relations", "type": "array", "default": None, "description": "Relation types to exclude"}
    ],
    
    "hardware_requirements": ["CPU required", "RAM: 4-16GB"],
    "time_complexity": "O(n) - Linear with number of triples",
    
    "outputs": [
        {"name": "dataset", "type": "torchdrug.data.KnowledgeGraphDataset", "description": "Knowledge graph dataset"},
        {"name": "num_entities", "type": "integer", "description": "Number of unique entities"},
        {"name": "num_relations", "type": "integer", "description": "Number of relation types"},
        {"name": "num_triples", "type": "integer", "description": "Number of triples"}
    ],
    
    "failure_modes": [
        {"condition": "Unknown dataset name", "behavior": "Raises ValueError"},
        {"condition": "Invalid relations filter", "behavior": "Raises KeyError"}
    ]
}

# =============================================================================
# MOLECULAR REPRESENTATION (3)
# =============================================================================

torchdrug_molecule_graph = {
    "name": "torchdrug_molecule_graph",
    "description": """Create molecular graph representations from SMILES.
        Constructs atom features, bond features, and adjacency.
        Supports various atom/bond featurization schemes.""",
    
    "required_parameters": [
        {"name": "smiles", "type": "string", "description": "SMILES string or list of SMILES"}
    ],
    
    "optional_parameters": [
        {"name": "atom_feature", "type": "string", "default": "default", "description": "Atom feature type (default, symbol, property_prediction)"},
        {"name": "bond_feature", "type": "string", "default": "default", "description": "Bond feature type (default, length)"},
        {"name": "with_hydrogen", "type": "boolean", "default": False, "description": "Include hydrogen atoms"},
        {"name": "kekulize", "type": "boolean", "default": False, "description": "Kekulize aromatic rings"}
    ],
    
    "hardware_requirements": ["CPU only", "RAM: < 2GB"],
    "time_complexity": "O(n) - ~5000 molecules/second",
    
    "outputs": [
        {"name": "graph", "type": "torchdrug.data.Molecule", "description": "Molecular graph object"},
        {"name": "node_feature", "type": "torch.Tensor", "description": "Node feature tensor"},
        {"name": "edge_feature", "type": "torch.Tensor", "description": "Edge feature tensor"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid SMILES", "behavior": "Raises ValueError"},
        {"condition": "RDKit parsing error", "behavior": "Raises RuntimeError"}
    ]
}

torchdrug_protein_graph = {
    "name": "torchdrug_protein_graph",
    "description": """Create protein graph representations from sequence or structure.
        Supports residue-level and atom-level graphs.
        Can use sequence features or 3D coordinates.""",
    
    "required_parameters": [
        {"name": "input", "type": "string", "description": "Protein sequence or PDB file path/ID"}
    ],
    
    "optional_parameters": [
        {"name": "graph_type", "type": "string", "default": "residue", "description": "Graph type (residue, atom)"},
        {"name": "edge_type", "type": "string", "default": "spatial", "description": "Edge construction (spatial, sequential, knn)"},
        {"name": "cutoff", "type": "number", "default": 8.0, "description": "Distance cutoff for spatial edges in Angstroms"},
        {"name": "atom_feature", "type": "string", "default": "default", "description": "Atom feature type"},
        {"name": "residue_feature", "type": "string", "default": "default", "description": "Residue feature type"}
    ],
    
    "hardware_requirements": ["CPU required", "RAM: 2-8GB"],
    "time_complexity": "O(n²) for spatial edges, O(n) for sequential",
    
    "outputs": [
        {"name": "graph", "type": "torchdrug.data.Protein", "description": "Protein graph object"},
        {"name": "num_residues", "type": "integer", "description": "Number of residues"},
        {"name": "node_feature", "type": "torch.Tensor", "description": "Node feature tensor"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid sequence", "behavior": "Raises ValueError"},
        {"condition": "PDB parsing error", "behavior": "Raises IOError"},
        {"condition": "Missing coordinates", "behavior": "Falls back to sequence-only mode"}
    ]
}

torchdrug_conformer_generation = {
    "name": "torchdrug_conformer_generation",
    "description": """Generate 3D conformers for molecules.
        Uses RDKit embedding with MMFF/UFF force field optimization.
        Returns lowest energy conformer(s).""",
    
    "required_parameters": [
        {"name": "smiles", "type": "string", "description": "SMILES string"}
    ],
    
    "optional_parameters": [
        {"name": "num_conformers", "type": "integer", "default": 10, "description": "Number of conformers to generate"},
        {"name": "force_field", "type": "string", "default": "MMFF", "description": "Force field (MMFF, UFF)"},
        {"name": "max_iterations", "type": "integer", "default": 200, "description": "Max optimization iterations"},
        {"name": "random_seed", "type": "integer", "default": None, "description": "Random seed for reproducibility"}
    ],
    
    "hardware_requirements": ["CPU only", "RAM: < 2GB"],
    "time_complexity": "O(n_conformers) - ~1-10 seconds per molecule",
    
    "outputs": [
        {"name": "conformers", "type": "list", "description": "List of 3D coordinate arrays"},
        {"name": "energies", "type": "list", "description": "Energy values for each conformer"},
        {"name": "best_conformer", "type": "numpy.ndarray", "description": "Lowest energy conformer coordinates"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid SMILES", "behavior": "Raises ValueError"},
        {"condition": "Embedding failure", "behavior": "Returns empty list"},
        {"condition": "Optimization failure", "behavior": "Returns unoptimized conformer"}
    ]
}

# =============================================================================
# GRAPH NEURAL NETWORK MODELS (4)
# =============================================================================

torchdrug_gin_model = {
    "name": "torchdrug_gin_model",
    "description": """Create a Graph Isomorphism Network for molecular property prediction.
        Implements the architecture from Xu et al. (2019).
        Theoretically most expressive message passing GNN.""",
    
    "required_parameters": [
        {"name": "input_dim", "type": "integer", "description": "Input node feature dimension"},
        {"name": "hidden_dims", "type": "array", "description": "Hidden layer dimensions list"}
    ],
    
    "optional_parameters": [
        {"name": "batch_norm", "type": "boolean", "default": True, "description": "Use batch normalization"},
        {"name": "activation", "type": "string", "default": "relu", "description": "Activation function"},
        {"name": "concat_hidden", "type": "boolean", "default": False, "description": "Concatenate all hidden layers"},
        {"name": "readout", "type": "string", "default": "sum", "description": "Graph readout (sum, mean, max)"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: 4-8GB", "RAM: 8GB"],
    "time_complexity": "Forward pass: O(n_nodes + n_edges) per graph",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.GIN", "description": "Initialized GIN model"},
        {"name": "output_dim", "type": "integer", "description": "Output embedding dimension"}
    ],
    
    "failure_modes": [
        {"condition": "Dimension mismatch", "behavior": "Raises ValueError"},
        {"condition": "Invalid activation", "behavior": "Raises KeyError"}
    ]
}

torchdrug_gcn_model = {
    "name": "torchdrug_gcn_model",
    "description": """Create a Graph Convolutional Network for molecular property prediction.
        Implements spectral graph convolutions from Kipf & Welling (2017).
        Standard baseline for molecular graph learning.""",
    
    "required_parameters": [
        {"name": "input_dim", "type": "integer", "description": "Input node feature dimension"},
        {"name": "hidden_dims", "type": "array", "description": "Hidden layer dimensions list"}
    ],
    
    "optional_parameters": [
        {"name": "batch_norm", "type": "boolean", "default": True, "description": "Use batch normalization"},
        {"name": "activation", "type": "string", "default": "relu", "description": "Activation function"},
        {"name": "dropout", "type": "number", "default": 0.0, "description": "Dropout rate"},
        {"name": "readout", "type": "string", "default": "sum", "description": "Graph readout method"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: 2-4GB", "RAM: 4GB"],
    "time_complexity": "Forward pass: O(n_nodes + n_edges) per graph",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.GCN", "description": "Initialized GCN model"}
    ],
    
    "failure_modes": [
        {"condition": "Dimension mismatch", "behavior": "Raises ValueError"}
    ]
}

torchdrug_schnet_model = {
    "name": "torchdrug_schnet_model",
    "description": """Create a SchNet model for 3D molecular property prediction.
        Uses continuous-filter convolutions on atomic distances.
        Designed for quantum chemistry predictions.""",
    
    "required_parameters": [
        {"name": "hidden_dim", "type": "integer", "description": "Hidden dimension size"}
    ],
    
    "optional_parameters": [
        {"name": "num_layers", "type": "integer", "default": 6, "description": "Number of interaction layers"},
        {"name": "num_filters", "type": "integer", "default": 128, "description": "Number of filters"},
        {"name": "num_gaussians", "type": "integer", "default": 50, "description": "Number of Gaussian basis functions"},
        {"name": "cutoff", "type": "number", "default": 10.0, "description": "Distance cutoff in Angstroms"},
        {"name": "readout", "type": "string", "default": "sum", "description": "Readout function"}
    ],
    
    "hardware_requirements": ["GPU required", "VRAM: 4-8GB", "RAM: 8GB"],
    "time_complexity": "O(n² * num_layers) for distance calculations",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.SchNet", "description": "Initialized SchNet model"}
    ],
    
    "failure_modes": [
        {"condition": "Missing 3D coordinates", "behavior": "Raises ValueError"},
        {"condition": "GPU OOM", "behavior": "Reduce num_filters or batch_size"}
    ]
}

torchdrug_gearnet_model = {
    "name": "torchdrug_gearnet_model",
    "description": """Create a GearNet model for protein representation learning.
        Multi-relational message passing on protein structures.
        State-of-the-art for protein function prediction.""",
    
    "required_parameters": [
        {"name": "input_dim", "type": "integer", "description": "Input node feature dimension"},
        {"name": "hidden_dims", "type": "array", "description": "Hidden layer dimensions list"}
    ],
    
    "optional_parameters": [
        {"name": "num_relation", "type": "integer", "default": 7, "description": "Number of edge relations"},
        {"name": "edge_input_dim", "type": "integer", "default": None, "description": "Edge feature dimension"},
        {"name": "batch_norm", "type": "boolean", "default": True, "description": "Use batch normalization"},
        {"name": "short_cut", "type": "boolean", "default": True, "description": "Use residual connections"},
        {"name": "readout", "type": "string", "default": "sum", "description": "Graph readout method"}
    ],
    
    "hardware_requirements": ["GPU required", "VRAM: 8-16GB", "RAM: 16GB"],
    "time_complexity": "O(n_residues * num_relation) per protein",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.GearNet", "description": "Initialized GearNet model"}
    ],
    
    "failure_modes": [
        {"condition": "Dimension mismatch", "behavior": "Raises ValueError"},
        {"condition": "Invalid num_relation", "behavior": "Raises ValueError"}
    ]
}

# =============================================================================
# GENERATIVE MODELS (3)
# =============================================================================

torchdrug_gcpn_model = {
    "name": "torchdrug_gcpn_model",
    "description": """Create a Graph Convolutional Policy Network for molecule generation.
        Reinforcement learning for goal-directed molecule design.
        Optimizes molecular properties through iterative graph construction.""",
    
    "required_parameters": [
        {"name": "atom_types", "type": "array", "description": "List of allowed atom types"},
        {"name": "hidden_dim", "type": "integer", "description": "Hidden dimension size"}
    ],
    
    "optional_parameters": [
        {"name": "num_layers", "type": "integer", "default": 3, "description": "Number of GNN layers"},
        {"name": "max_edge_unroll", "type": "integer", "default": 12, "description": "Max edges per step"},
        {"name": "gamma", "type": "number", "default": 0.99, "description": "Reward discount factor"},
        {"name": "learning_rate", "type": "number", "default": 1e-4, "description": "Learning rate"}
    ],
    
    "hardware_requirements": ["GPU required", "VRAM: 8-16GB", "RAM: 16GB"],
    "time_complexity": "Training: ~hours for convergence",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.GCPN", "description": "Initialized GCPN model"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid atom types", "behavior": "Raises ValueError"},
        {"condition": "Training instability", "behavior": "Reduce learning_rate"}
    ]
}

torchdrug_graphaf_model = {
    "name": "torchdrug_graphaf_model",
    "description": """Create a GraphAF model for autoregressive molecule generation.
        Flow-based generative model for molecular graphs.
        Generates valid molecules by construction.""",
    
    "required_parameters": [
        {"name": "atom_types", "type": "array", "description": "List of allowed atom types"},
        {"name": "hidden_dim", "type": "integer", "description": "Hidden dimension size"}
    ],
    
    "optional_parameters": [
        {"name": "num_layers", "type": "integer", "default": 3, "description": "Number of GNN layers"},
        {"name": "num_flow_layers", "type": "integer", "default": 6, "description": "Number of flow layers"},
        {"name": "max_node", "type": "integer", "default": 38, "description": "Maximum number of atoms"},
        {"name": "learning_rate", "type": "number", "default": 1e-3, "description": "Learning rate"}
    ],
    
    "hardware_requirements": ["GPU required", "VRAM: 8-16GB", "RAM: 16GB"],
    "time_complexity": "Generation: ~0.1 sec/molecule",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.GraphAF", "description": "Initialized GraphAF model"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid atom types", "behavior": "Raises ValueError"},
        {"condition": "Exceeds max_node", "behavior": "Truncates generation"}
    ]
}

torchdrug_vae_model = {
    "name": "torchdrug_vae_model",
    "description": """Create a Variational Autoencoder for molecular generation.
        Latent space model for molecular optimization.
        Supports property-guided generation via latent interpolation.""",
    
    "required_parameters": [
        {"name": "encoder", "type": "object", "description": "Encoder model (GNN)"},
        {"name": "decoder", "type": "object", "description": "Decoder model (RNN or GNN)"},
        {"name": "latent_dim", "type": "integer", "description": "Latent space dimension"}
    ],
    
    "optional_parameters": [
        {"name": "beta", "type": "number", "default": 1.0, "description": "KL divergence weight"},
        {"name": "learning_rate", "type": "number", "default": 1e-3, "description": "Learning rate"}
    ],
    
    "hardware_requirements": ["GPU required", "VRAM: 4-8GB", "RAM: 8GB"],
    "time_complexity": "Training: O(epochs * n_molecules)",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.VariationalAutoEncoder", "description": "Initialized VAE model"}
    ],
    
    "failure_modes": [
        {"condition": "Encoder/decoder mismatch", "behavior": "Raises ValueError"},
        {"condition": "KL collapse", "behavior": "Reduce beta parameter"}
    ]
}

# =============================================================================
# KNOWLEDGE GRAPH MODELS (2)
# =============================================================================

torchdrug_rotate_model = {
    "name": "torchdrug_rotate_model",
    "description": """Create a RotatE model for knowledge graph completion.
        Rotation-based embedding for relation modeling.
        Captures symmetry, antisymmetry, and composition patterns.""",
    
    "required_parameters": [
        {"name": "num_entity", "type": "integer", "description": "Number of entities"},
        {"name": "num_relation", "type": "integer", "description": "Number of relations"},
        {"name": "embedding_dim", "type": "integer", "description": "Embedding dimension (must be even)"}
    ],
    
    "optional_parameters": [
        {"name": "gamma", "type": "number", "default": 12.0, "description": "Margin parameter"},
        {"name": "negative_samples", "type": "integer", "default": 256, "description": "Negative samples per positive"},
        {"name": "learning_rate", "type": "number", "default": 1e-4, "description": "Learning rate"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: 4-16GB depending on KG size", "RAM: 8-32GB"],
    "time_complexity": "Training: O(epochs * n_triples * negative_samples)",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.RotatE", "description": "Initialized RotatE model"}
    ],
    
    "failure_modes": [
        {"condition": "Odd embedding_dim", "behavior": "Raises ValueError"},
        {"condition": "Invalid entity/relation indices", "behavior": "Raises IndexError"}
    ]
}

torchdrug_nbt_model = {
    "name": "torchdrug_nbt_model",
    "description": """Create a Neural Bellman-Ford Network for knowledge graph reasoning.
        Path-based reasoning with neural message passing.
        Supports multi-hop inference and rule learning.""",
    
    "required_parameters": [
        {"name": "num_relation", "type": "integer", "description": "Number of relations"},
        {"name": "hidden_dim", "type": "integer", "description": "Hidden dimension size"}
    ],
    
    "optional_parameters": [
        {"name": "num_layers", "type": "integer", "default": 3, "description": "Number of reasoning layers"},
        {"name": "num_mlp_layers", "type": "integer", "default": 2, "description": "MLP layers per reasoning step"},
        {"name": "learning_rate", "type": "number", "default": 1e-3, "description": "Learning rate"}
    ],
    
    "hardware_requirements": ["GPU required", "VRAM: 8-32GB", "RAM: 16-64GB"],
    "time_complexity": "Inference: O(n_entities * num_layers)",
    
    "outputs": [
        {"name": "model", "type": "torchdrug.models.NBFNet", "description": "Initialized NBFNet model"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid num_relation", "behavior": "Raises ValueError"},
        {"condition": "GPU OOM", "behavior": "Reduce hidden_dim or batch_size"}
    ]
}

# =============================================================================
# TRAINING & TASKS (3)
# =============================================================================

torchdrug_property_prediction_task = {
    "name": "torchdrug_property_prediction_task",
    "description": """Create a property prediction task wrapper.
        Combines GNN encoder with prediction head.
        Handles classification and regression tasks.""",
    
    "required_parameters": [
        {"name": "model", "type": "object", "description": "GNN encoder model"},
        {"name": "task_type", "type": "string", "description": "Task type (classification, regression)"},
        {"name": "num_tasks", "type": "integer", "description": "Number of prediction targets"}
    ],
    
    "optional_parameters": [
        {"name": "num_classes", "type": "integer", "default": 2, "description": "Classes per task (for classification)"},
        {"name": "metric", "type": "array", "default": ["auc"], "description": "Evaluation metrics"},
        {"name": "normalization", "type": "boolean", "default": True, "description": "Normalize graph features"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: 4-8GB", "RAM: 8GB"],
    "time_complexity": "Depends on underlying model",
    
    "outputs": [
        {"name": "task", "type": "torchdrug.tasks.PropertyPrediction", "description": "Property prediction task"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid task_type", "behavior": "Raises ValueError"},
        {"condition": "Model incompatibility", "behavior": "Raises RuntimeError"}
    ]
}

torchdrug_generation_task = {
    "name": "torchdrug_generation_task",
    "description": """Create a molecule generation task wrapper.
        Handles training loop for generative models.
        Supports property-guided generation.""",
    
    "required_parameters": [
        {"name": "model", "type": "object", "description": "Generative model (GCPN, GraphAF, etc.)"}
    ],
    
    "optional_parameters": [
        {"name": "reward_functions", "type": "array", "default": [], "description": "Property reward functions"},
        {"name": "reward_weights", "type": "array", "default": [], "description": "Weights for each reward"},
        {"name": "max_epoch", "type": "integer", "default": 100, "description": "Maximum training epochs"}
    ],
    
    "hardware_requirements": ["GPU required", "VRAM: 8-16GB", "RAM: 16GB"],
    "time_complexity": "Training: ~hours",
    
    "outputs": [
        {"name": "task", "type": "torchdrug.tasks.Generation", "description": "Generation task"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid reward function", "behavior": "Raises ValueError"},
        {"condition": "Reward weight mismatch", "behavior": "Raises ValueError"}
    ]
}

torchdrug_train_and_evaluate = {
    "name": "torchdrug_train_and_evaluate",
    "description": """Train and evaluate TorchDrug models with standardized pipeline.
        Handles data loading, training loop, and metric computation.
        Supports early stopping and model checkpointing.""",
    
    "required_parameters": [
        {"name": "task", "type": "object", "description": "TorchDrug task object"},
        {"name": "train_set", "type": "object", "description": "Training dataset"},
        {"name": "valid_set", "type": "object", "description": "Validation dataset"},
        {"name": "test_set", "type": "object", "description": "Test dataset"}
    ],
    
    "optional_parameters": [
        {"name": "batch_size", "type": "integer", "default": 32, "description": "Training batch size"},
        {"name": "num_epoch", "type": "integer", "default": 100, "description": "Number of epochs"},
        {"name": "patience", "type": "integer", "default": 10, "description": "Early stopping patience"},
        {"name": "learning_rate", "type": "number", "default": 1e-3, "description": "Learning rate"},
        {"name": "checkpoint_dir", "type": "string", "default": None, "description": "Checkpoint directory"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: Depends on model", "RAM: 8-16GB"],
    "time_complexity": "O(epochs * dataset_size / batch_size)",
    
    "outputs": [
        {"name": "train_metrics", "type": "dict", "description": "Final training metrics"},
        {"name": "valid_metrics", "type": "dict", "description": "Best validation metrics"},
        {"name": "test_metrics", "type": "dict", "description": "Test set metrics"}
    ],
    
    "failure_modes": [
        {"condition": "Dataset/task mismatch", "behavior": "Raises RuntimeError"},
        {"condition": "Training divergence", "behavior": "Returns NaN metrics"}
    ]
}

# =============================================================================
# UTILITIES (2)
# =============================================================================

torchdrug_molecular_property = {
    "name": "torchdrug_molecular_property",
    "description": """Calculate molecular properties using RDKit via TorchDrug.
        Computes drug-likeness metrics, physicochemical properties, and ADMET-related descriptors.""",
    
    "required_parameters": [
        {"name": "molecule", "type": "string", "description": "SMILES string or TorchDrug Molecule object"}
    ],
    
    "optional_parameters": [
        {"name": "properties", "type": "array", "default": ["all"], "description": "List of properties to calculate (QED, SA, LogP, MW, TPSA, etc.)"}
    ],
    
    "hardware_requirements": ["CPU only", "RAM: < 1GB"],
    "time_complexity": "O(1) - ~1000 molecules/second",
    
    "outputs": [
        {"name": "properties", "type": "dict", "description": "Dictionary of property names to values"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid SMILES", "behavior": "Raises ValueError"},
        {"condition": "Unknown property", "behavior": "Raises KeyError"}
    ]
}

torchdrug_similarity_search = {
    "name": "torchdrug_similarity_search",
    "description": """Perform molecular similarity search using fingerprints or learned embeddings.
        Finds nearest neighbors in chemical space.
        Supports Tanimoto, cosine, and Euclidean distances.""",
    
    "required_parameters": [
        {"name": "query", "type": "string", "description": "Query SMILES or embedding"},
        {"name": "database", "type": "array", "description": "List of SMILES or embeddings to search"}
    ],
    
    "optional_parameters": [
        {"name": "method", "type": "string", "default": "tanimoto", "description": "Similarity method (tanimoto, cosine, euclidean)"},
        {"name": "top_k", "type": "integer", "default": 10, "description": "Number of results to return"},
        {"name": "fingerprint", "type": "string", "default": "morgan", "description": "Fingerprint type if using SMILES"}
    ],
    
    "hardware_requirements": ["CPU required", "GPU optional for embeddings", "RAM: 2-8GB"],
    "time_complexity": "O(n * d) for database of n molecules with d-dimensional features",
    
    "outputs": [
        {"name": "indices", "type": "list", "description": "Indices of top-k similar molecules"},
        {"name": "similarities", "type": "list", "description": "Similarity scores"},
        {"name": "molecules", "type": "list", "description": "Top-k similar SMILES strings"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid query SMILES", "behavior": "Raises ValueError"},
        {"condition": "Empty database", "behavior": "Returns empty results"}
    ]
}

# =============================================================================
# COLLECTION OF ALL TORCHDRUG TOOLS
# =============================================================================

TORCHDRUG_TOOLS = [
    # Data loading
    torchdrug_load_molecule_dataset,
    torchdrug_load_protein_dataset,
    torchdrug_load_knowledge_graph,
    # Molecular representation
    torchdrug_molecule_graph,
    torchdrug_protein_graph,
    torchdrug_conformer_generation,
    # GNN Models
    torchdrug_gin_model,
    torchdrug_gcn_model,
    torchdrug_schnet_model,
    torchdrug_gearnet_model,
    # Generative Models
    torchdrug_gcpn_model,
    torchdrug_graphaf_model,
    torchdrug_vae_model,
    # Knowledge Graph Models
    torchdrug_rotate_model,
    torchdrug_nbt_model,
    # Training & Tasks
    torchdrug_property_prediction_task,
    torchdrug_generation_task,
    torchdrug_train_and_evaluate,
    # Utilities
    torchdrug_molecular_property,
    torchdrug_similarity_search,
]


def get_torchdrug_tools_for_openai() -> list:
    """Get all TorchDrug tools in OpenAI function calling format."""
    from template import get_openai_function_schema
    return [get_openai_function_schema(tool) for tool in TORCHDRUG_TOOLS]


def get_torchdrug_tools_for_anthropic() -> list:
    """Get all TorchDrug tools in Anthropic tool use format."""
    from template import get_anthropic_tool_schema
    return [get_anthropic_tool_schema(tool) for tool in TORCHDRUG_TOOLS]


if __name__ == "__main__":
    print(f"TorchDrug Tools: {len(TORCHDRUG_TOOLS)} tools defined")
    for tool in TORCHDRUG_TOOLS:
        print(f"  - {tool['name']}")
