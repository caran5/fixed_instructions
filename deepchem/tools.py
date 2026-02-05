"""
DeepChem Tool Descriptions for Function Calling

Comprehensive tool definitions for molecular machine learning with DeepChem.
Covers data loading, featurization, model training, and prediction.
"""

# =============================================================================
# DATA LOADING (2)
# =============================================================================

deepchem_load_data = {
    "name": "deepchem_load_data",
    "description": """Load molecular data from CSV or SDF files into DeepChem Dataset objects.
        Supports SMILES strings and SDF molecular structures. Handles multi-task datasets
        with automatic featurization. Returns featurized DiskDataset ready for training.""",
    
    "required_parameters": [
        {"name": "input_file", "type": "string", "description": "Path to CSV or SDF file containing molecular data"},
        {"name": "smiles_field", "type": "string", "description": "Column name containing SMILES strings (for CSV)"}
    ],
    
    "optional_parameters": [
        {"name": "tasks", "type": "array", "default": None, "description": "List of column names for prediction targets"},
        {"name": "featurizer", "type": "string", "default": "ECFP", "description": "Featurizer to use (ECFP, GraphConv, Weave, etc.)"},
        {"name": "id_field", "type": "string", "default": None, "description": "Column name for molecule IDs"},
        {"name": "shard_size", "type": "integer", "default": 8192, "description": "Number of molecules per shard"}
    ],
    
    "hardware_requirements": ["CPU required", "GPU optional", "RAM: 4GB minimum, 16GB recommended for large datasets"],
    "time_complexity": "O(n) - Linear with number of molecules, ~1000 mol/sec for ECFP",
    
    "outputs": [
        {"name": "dataset", "type": "dc.data.DiskDataset", "description": "Featurized dataset ready for training"},
        {"name": "tasks", "type": "list", "description": "List of task names"},
        {"name": "transformers", "type": "list", "description": "List of applied transformers"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid SMILES", "behavior": "Skips invalid molecules, logs warning"},
        {"condition": "Missing columns", "behavior": "Raises KeyError"},
        {"condition": "Memory overflow", "behavior": "Use smaller shard_size"}
    ]
}

deepchem_load_molnet = {
    "name": "deepchem_load_molnet",
    "description": """Load MoleculeNet benchmark datasets with standardized splits and featurization.
        Provides access to 17+ curated datasets for molecular property prediction including
        Tox21, BBBP, HIV, ESOL, FreeSolv, Lipophilicity, and more.""",
    
    "required_parameters": [
        {"name": "dataset_name", "type": "string", "description": "MoleculeNet dataset name (tox21, bbbp, hiv, esol, freesolv, etc.)"}
    ],
    
    "optional_parameters": [
        {"name": "featurizer", "type": "string", "default": "ECFP", "description": "Featurizer type (ECFP, GraphConv, Weave, etc.)"},
        {"name": "splitter", "type": "string", "default": "scaffold", "description": "Data splitter (scaffold, random, stratified)"},
        {"name": "reload", "type": "boolean", "default": True, "description": "Whether to reload cached dataset"},
        {"name": "transformers", "type": "array", "default": ["NormalizationTransformer"], "description": "List of transformers to apply"}
    ],
    
    "hardware_requirements": ["CPU required", "GPU optional", "RAM: 2-8GB depending on dataset"],
    "time_complexity": "O(1) - Cached datasets load in seconds",
    
    "outputs": [
        {"name": "tasks", "type": "list", "description": "List of prediction task names"},
        {"name": "datasets", "type": "tuple", "description": "(train, valid, test) Dataset tuple"},
        {"name": "transformers", "type": "list", "description": "Applied transformers"}
    ],
    
    "failure_modes": [
        {"condition": "Unknown dataset name", "behavior": "Raises ValueError with valid options"},
        {"condition": "Download failure", "behavior": "Raises ConnectionError"}
    ]
}

# =============================================================================
# FEATURIZATION (4)
# =============================================================================

deepchem_circular_fingerprint = {
    "name": "deepchem_circular_fingerprint",
    "description": """Generate Extended Connectivity Fingerprints (ECFP) from SMILES strings.
        Produces fixed-length binary vectors encoding molecular substructures.
        Standard representation for molecular similarity and ML models.""",
    
    "required_parameters": [
        {"name": "smiles", "type": "string", "description": "SMILES string or list of SMILES strings"}
    ],
    
    "optional_parameters": [
        {"name": "radius", "type": "integer", "default": 2, "description": "Fingerprint radius (2=ECFP4, 3=ECFP6)"},
        {"name": "size", "type": "integer", "default": 2048, "description": "Fingerprint bit vector length"},
        {"name": "chiral", "type": "boolean", "default": False, "description": "Include chirality in fingerprint"},
        {"name": "bonds", "type": "boolean", "default": True, "description": "Include bond information"},
        {"name": "features", "type": "boolean", "default": True, "description": "Use feature invariants"}
    ],
    
    "hardware_requirements": ["CPU only", "RAM: < 1GB"],
    "time_complexity": "O(n) - ~10,000 molecules/second",
    
    "outputs": [
        {"name": "fingerprints", "type": "numpy.ndarray", "description": "Binary fingerprint array of shape (n_molecules, size)"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid SMILES", "behavior": "Returns None for that molecule"},
        {"condition": "Empty input", "behavior": "Returns empty array"}
    ]
}

deepchem_rdkit_descriptors = {
    "name": "deepchem_rdkit_descriptors",
    "description": """Calculate RDKit molecular descriptors from SMILES strings.
        Computes 200+ physicochemical descriptors including molecular weight,
        LogP, TPSA, rotatable bonds, H-bond donors/acceptors, and more.""",
    
    "required_parameters": [
        {"name": "smiles", "type": "string", "description": "SMILES string or list of SMILES strings"}
    ],
    
    "optional_parameters": [
        {"name": "descriptors", "type": "array", "default": None, "description": "List of specific descriptors to calculate (default: all)"},
        {"name": "is_normalized", "type": "boolean", "default": False, "description": "Whether to normalize descriptor values"}
    ],
    
    "hardware_requirements": ["CPU only", "RAM: < 1GB"],
    "time_complexity": "O(n) - ~1,000 molecules/second",
    
    "outputs": [
        {"name": "descriptors", "type": "numpy.ndarray", "description": "Descriptor array of shape (n_molecules, n_descriptors)"},
        {"name": "descriptor_names", "type": "list", "description": "Names of computed descriptors"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid SMILES", "behavior": "Returns NaN for that molecule"},
        {"condition": "Unknown descriptor name", "behavior": "Raises KeyError"}
    ]
}

deepchem_mol_graph_conv_featurizer = {
    "name": "deepchem_mol_graph_conv_featurizer",
    "description": """Convert molecules to graph representations for Graph Neural Networks.
        Creates node features (atoms), edge features (bonds), and adjacency matrices.
        Compatible with GCN, GAT, MPNN, and other graph-based models.""",
    
    "required_parameters": [
        {"name": "smiles", "type": "string", "description": "SMILES string or list of SMILES strings"}
    ],
    
    "optional_parameters": [
        {"name": "use_edges", "type": "boolean", "default": True, "description": "Include edge/bond features"},
        {"name": "use_chirality", "type": "boolean", "default": False, "description": "Include chirality features"},
        {"name": "use_partial_charge", "type": "boolean", "default": False, "description": "Include Gasteiger partial charges"}
    ],
    
    "hardware_requirements": ["CPU required", "GPU optional for model training", "RAM: 2-4GB"],
    "time_complexity": "O(n) - ~500 molecules/second",
    
    "outputs": [
        {"name": "graphs", "type": "list[ConvMol]", "description": "List of ConvMol graph objects"},
        {"name": "node_features", "type": "numpy.ndarray", "description": "Atom feature matrices"},
        {"name": "adjacency", "type": "numpy.ndarray", "description": "Adjacency matrices"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid SMILES", "behavior": "Returns None for that molecule"},
        {"condition": "Empty molecule", "behavior": "Skips with warning"}
    ]
}

deepchem_featurize_molecules = {
    "name": "deepchem_featurize_molecules",
    "description": """General-purpose molecular featurization wrapper.
        Supports multiple featurizer types with unified interface.
        Handles batch processing and error recovery.""",
    
    "required_parameters": [
        {"name": "molecules", "type": "array", "description": "List of SMILES strings or RDKit Mol objects"},
        {"name": "featurizer_type", "type": "string", "description": "Type of featurizer (ECFP, GraphConv, Weave, Coulomb, etc.)"}
    ],
    
    "optional_parameters": [
        {"name": "featurizer_params", "type": "object", "default": {}, "description": "Dictionary of featurizer-specific parameters"},
        {"name": "log_every_n", "type": "integer", "default": 1000, "description": "Log progress every N molecules"}
    ],
    
    "hardware_requirements": ["CPU required", "GPU optional", "RAM: 2-8GB depending on featurizer"],
    "time_complexity": "O(n) - Varies by featurizer type",
    
    "outputs": [
        {"name": "features", "type": "numpy.ndarray", "description": "Feature array or list of graph objects"},
        {"name": "valid_indices", "type": "list", "description": "Indices of successfully featurized molecules"}
    ],
    
    "failure_modes": [
        {"condition": "Unknown featurizer type", "behavior": "Raises ValueError"},
        {"condition": "Batch failures", "behavior": "Returns partial results with invalid indices"}
    ]
}

# =============================================================================
# DATA SPLITTING (1)
# =============================================================================

deepchem_scaffold_splitter = {
    "name": "deepchem_scaffold_splitter",
    "description": """Split molecular dataset by Murcko scaffolds for realistic train/test evaluation.
        Groups molecules by common core structures to prevent data leakage.
        Standard practice for molecular property prediction benchmarks.""",
    
    "required_parameters": [
        {"name": "dataset", "type": "dc.data.Dataset", "description": "DeepChem Dataset object to split"}
    ],
    
    "optional_parameters": [
        {"name": "frac_train", "type": "number", "default": 0.8, "description": "Fraction for training set"},
        {"name": "frac_valid", "type": "number", "default": 0.1, "description": "Fraction for validation set"},
        {"name": "frac_test", "type": "number", "default": 0.1, "description": "Fraction for test set"},
        {"name": "seed", "type": "integer", "default": None, "description": "Random seed for reproducibility"}
    ],
    
    "hardware_requirements": ["CPU only", "RAM: 1-2GB"],
    "time_complexity": "O(n log n) - Scaffold extraction and sorting",
    
    "outputs": [
        {"name": "train_dataset", "type": "dc.data.Dataset", "description": "Training split"},
        {"name": "valid_dataset", "type": "dc.data.Dataset", "description": "Validation split"},
        {"name": "test_dataset", "type": "dc.data.Dataset", "description": "Test split"}
    ],
    
    "failure_modes": [
        {"condition": "Fractions don't sum to 1", "behavior": "Raises ValueError"},
        {"condition": "Too few scaffolds", "behavior": "Warning, uneven splits"}
    ]
}

# =============================================================================
# MODELS (6)
# =============================================================================

deepchem_gcn_model = {
    "name": "deepchem_gcn_model",
    "description": """Create a Graph Convolutional Network model for molecular property prediction.
        Implements the architecture from Kipf & Welling (2017).
        Supports classification and regression tasks.""",
    
    "required_parameters": [
        {"name": "n_tasks", "type": "integer", "description": "Number of prediction tasks"},
        {"name": "mode", "type": "string", "description": "Task type: 'classification' or 'regression'"}
    ],
    
    "optional_parameters": [
        {"name": "graph_conv_layers", "type": "array", "default": [64, 64], "description": "Hidden units per graph conv layer"},
        {"name": "dense_layers", "type": "array", "default": [128], "description": "Hidden units per dense layer"},
        {"name": "dropout", "type": "number", "default": 0.0, "description": "Dropout rate"},
        {"name": "learning_rate", "type": "number", "default": 0.001, "description": "Adam optimizer learning rate"},
        {"name": "batch_size", "type": "integer", "default": 100, "description": "Training batch size"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: 4-8GB", "RAM: 8GB"],
    "time_complexity": "Training: O(epochs * n_molecules), ~1-2 min/epoch for 10K molecules on GPU",
    
    "outputs": [
        {"name": "model", "type": "dc.models.GCNModel", "description": "Initialized GCN model"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid mode", "behavior": "Raises ValueError"},
        {"condition": "GPU OOM", "behavior": "Reduce batch_size"}
    ]
}

deepchem_gat_model = {
    "name": "deepchem_gat_model",
    "description": """Create a Graph Attention Network model for molecular property prediction.
        Implements multi-head attention mechanism from Veličković et al. (2018).
        Often outperforms GCN on molecular tasks.""",
    
    "required_parameters": [
        {"name": "n_tasks", "type": "integer", "description": "Number of prediction tasks"},
        {"name": "mode", "type": "string", "description": "Task type: 'classification' or 'regression'"}
    ],
    
    "optional_parameters": [
        {"name": "n_attention_heads", "type": "integer", "default": 8, "description": "Number of attention heads"},
        {"name": "graph_attention_layers", "type": "array", "default": [64, 64], "description": "Hidden units per GAT layer"},
        {"name": "dense_layers", "type": "array", "default": [128], "description": "Hidden units per dense layer"},
        {"name": "dropout", "type": "number", "default": 0.0, "description": "Dropout rate"},
        {"name": "learning_rate", "type": "number", "default": 0.001, "description": "Learning rate"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: 6-12GB", "RAM: 8GB"],
    "time_complexity": "Training: O(epochs * n_molecules * n_heads), ~2-3 min/epoch for 10K molecules",
    
    "outputs": [
        {"name": "model", "type": "dc.models.GATModel", "description": "Initialized GAT model"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid mode", "behavior": "Raises ValueError"},
        {"condition": "GPU OOM", "behavior": "Reduce n_attention_heads or batch_size"}
    ]
}

deepchem_attentive_fp_model = {
    "name": "deepchem_attentive_fp_model",
    "description": """Create an Attentive Fingerprint model for molecular property prediction.
        Implements graph attention with learned molecular fingerprints.
        State-of-the-art for many molecular property prediction tasks.""",
    
    "required_parameters": [
        {"name": "n_tasks", "type": "integer", "description": "Number of prediction tasks"},
        {"name": "mode", "type": "string", "description": "Task type: 'classification' or 'regression'"}
    ],
    
    "optional_parameters": [
        {"name": "num_layers", "type": "integer", "default": 2, "description": "Number of graph attention layers"},
        {"name": "num_timesteps", "type": "integer", "default": 2, "description": "Number of timesteps for attention"},
        {"name": "graph_feat_size", "type": "integer", "default": 200, "description": "Graph feature size"},
        {"name": "dropout", "type": "number", "default": 0.0, "description": "Dropout rate"},
        {"name": "learning_rate", "type": "number", "default": 0.001, "description": "Learning rate"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: 4-8GB", "RAM: 8GB"],
    "time_complexity": "Training: ~2-4 min/epoch for 10K molecules on GPU",
    
    "outputs": [
        {"name": "model", "type": "dc.models.AttentiveFPModel", "description": "Initialized AttentiveFP model"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid mode", "behavior": "Raises ValueError"},
        {"condition": "GPU OOM", "behavior": "Reduce graph_feat_size"}
    ]
}

deepchem_multitask_classifier = {
    "name": "deepchem_multitask_classifier",
    "description": """Create a multi-task dense neural network classifier.
        Feedforward network with shared layers for multiple classification tasks.
        Suitable for fingerprint-based molecular classification.""",
    
    "required_parameters": [
        {"name": "n_tasks", "type": "integer", "description": "Number of classification tasks"},
        {"name": "n_features", "type": "integer", "description": "Input feature dimension"}
    ],
    
    "optional_parameters": [
        {"name": "layer_sizes", "type": "array", "default": [1000], "description": "Hidden layer sizes"},
        {"name": "dropout", "type": "number", "default": 0.5, "description": "Dropout rate"},
        {"name": "learning_rate", "type": "number", "default": 0.001, "description": "Learning rate"},
        {"name": "weight_decay", "type": "number", "default": 0.0, "description": "L2 regularization"},
        {"name": "batch_size", "type": "integer", "default": 50, "description": "Training batch size"}
    ],
    
    "hardware_requirements": ["CPU or GPU", "VRAM: 2-4GB if GPU", "RAM: 4GB"],
    "time_complexity": "Training: ~30 sec/epoch for 10K molecules",
    
    "outputs": [
        {"name": "model", "type": "dc.models.MultitaskClassifier", "description": "Initialized classifier"}
    ],
    
    "failure_modes": [
        {"condition": "Feature dimension mismatch", "behavior": "Raises ValueError"},
        {"condition": "Invalid n_tasks", "behavior": "Raises ValueError"}
    ]
}

deepchem_multitask_regressor = {
    "name": "deepchem_multitask_regressor",
    "description": """Create a multi-task dense neural network regressor.
        Feedforward network with shared layers for multiple regression tasks.
        Suitable for continuous property prediction.""",
    
    "required_parameters": [
        {"name": "n_tasks", "type": "integer", "description": "Number of regression tasks"},
        {"name": "n_features", "type": "integer", "description": "Input feature dimension"}
    ],
    
    "optional_parameters": [
        {"name": "layer_sizes", "type": "array", "default": [1000], "description": "Hidden layer sizes"},
        {"name": "dropout", "type": "number", "default": 0.5, "description": "Dropout rate"},
        {"name": "learning_rate", "type": "number", "default": 0.001, "description": "Learning rate"},
        {"name": "weight_decay", "type": "number", "default": 0.0, "description": "L2 regularization"},
        {"name": "batch_size", "type": "integer", "default": 50, "description": "Training batch size"}
    ],
    
    "hardware_requirements": ["CPU or GPU", "VRAM: 2-4GB if GPU", "RAM: 4GB"],
    "time_complexity": "Training: ~30 sec/epoch for 10K molecules",
    
    "outputs": [
        {"name": "model", "type": "dc.models.MultitaskRegressor", "description": "Initialized regressor"}
    ],
    
    "failure_modes": [
        {"condition": "Feature dimension mismatch", "behavior": "Raises ValueError"},
        {"condition": "Invalid n_tasks", "behavior": "Raises ValueError"}
    ]
}

deepchem_huggingface_model = {
    "name": "deepchem_huggingface_model",
    "description": """Create a HuggingFace transformer model for molecular property prediction.
        Supports ChemBERTa, MoLFormer, and other pretrained chemical language models.
        Enables transfer learning from large-scale pretraining.""",
    
    "required_parameters": [
        {"name": "task", "type": "string", "description": "Task name matching HuggingFace model head"},
        {"name": "model_name", "type": "string", "description": "HuggingFace model identifier (e.g., 'seyonec/ChemBERTa-zinc-base-v1')"}
    ],
    
    "optional_parameters": [
        {"name": "n_tasks", "type": "integer", "default": 1, "description": "Number of prediction tasks"},
        {"name": "mode", "type": "string", "default": "regression", "description": "Task type: 'classification' or 'regression'"},
        {"name": "learning_rate", "type": "number", "default": 5e-5, "description": "Learning rate for fine-tuning"},
        {"name": "batch_size", "type": "integer", "default": 8, "description": "Training batch size"}
    ],
    
    "hardware_requirements": ["GPU required", "VRAM: 8-16GB", "RAM: 16GB"],
    "time_complexity": "Training: ~5-10 min/epoch depending on model size",
    
    "outputs": [
        {"name": "model", "type": "dc.models.HuggingFaceModel", "description": "Initialized HuggingFace model"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid model name", "behavior": "Raises OSError"},
        {"condition": "GPU OOM", "behavior": "Reduce batch_size or use gradient accumulation"}
    ]
}

# =============================================================================
# TRAINING & EVALUATION (3)
# =============================================================================

deepchem_train_graph_model = {
    "name": "deepchem_train_graph_model",
    "description": """Train a graph neural network model on molecular data.
        Handles training loop with validation monitoring and early stopping.
        Supports checkpointing and learning rate scheduling.""",
    
    "required_parameters": [
        {"name": "model", "type": "dc.models.Model", "description": "DeepChem model to train"},
        {"name": "train_dataset", "type": "dc.data.Dataset", "description": "Training dataset"},
        {"name": "valid_dataset", "type": "dc.data.Dataset", "description": "Validation dataset"}
    ],
    
    "optional_parameters": [
        {"name": "nb_epoch", "type": "integer", "default": 100, "description": "Maximum number of epochs"},
        {"name": "patience", "type": "integer", "default": 10, "description": "Early stopping patience"},
        {"name": "checkpoint_interval", "type": "integer", "default": 10, "description": "Epochs between checkpoints"},
        {"name": "model_dir", "type": "string", "default": None, "description": "Directory for model checkpoints"}
    ],
    
    "hardware_requirements": ["GPU recommended", "VRAM: 4-16GB", "RAM: 8-16GB"],
    "time_complexity": "O(epochs * n_molecules) - Highly variable",
    
    "outputs": [
        {"name": "loss_history", "type": "list", "description": "Training loss per epoch"},
        {"name": "valid_scores", "type": "list", "description": "Validation metrics per epoch"},
        {"name": "best_epoch", "type": "integer", "description": "Epoch with best validation score"}
    ],
    
    "failure_modes": [
        {"condition": "Dataset/model mismatch", "behavior": "Raises ValueError"},
        {"condition": "Training divergence", "behavior": "Returns NaN losses"}
    ]
}

deepchem_evaluate_model = {
    "name": "deepchem_evaluate_model",
    "description": """Evaluate a trained DeepChem model on a test dataset.
        Computes standard metrics (AUC-ROC, accuracy, R², MAE, etc.) based on task type.
        Supports multi-task evaluation with per-task metrics.""",
    
    "required_parameters": [
        {"name": "model", "type": "dc.models.Model", "description": "Trained DeepChem model"},
        {"name": "dataset", "type": "dc.data.Dataset", "description": "Dataset to evaluate on"},
        {"name": "metrics", "type": "array", "description": "List of metric names (roc_auc, accuracy, r2, mae, rmse, etc.)"}
    ],
    
    "optional_parameters": [
        {"name": "transformers", "type": "array", "default": [], "description": "Transformers to apply before evaluation"},
        {"name": "per_task_metrics", "type": "boolean", "default": True, "description": "Return metrics per task"}
    ],
    
    "hardware_requirements": ["CPU or GPU", "RAM: 4GB"],
    "time_complexity": "O(n) - Linear with dataset size",
    
    "outputs": [
        {"name": "scores", "type": "dict", "description": "Dictionary of metric names to scores"},
        {"name": "per_task_scores", "type": "dict", "description": "Per-task metric breakdown"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid metric name", "behavior": "Raises ValueError"},
        {"condition": "Untrained model", "behavior": "Returns random/poor scores"}
    ]
}

deepchem_predict = {
    "name": "deepchem_predict",
    "description": """Make predictions on new molecules using a trained DeepChem model.
        Handles featurization and returns predictions with optional uncertainties.""",
    
    "required_parameters": [
        {"name": "model", "type": "dc.models.Model", "description": "Trained DeepChem model"},
        {"name": "molecules", "type": "array", "description": "SMILES strings or featurized data"}
    ],
    
    "optional_parameters": [
        {"name": "transformers", "type": "array", "default": [], "description": "Transformers for inverse transformation"},
        {"name": "uncertainty", "type": "boolean", "default": False, "description": "Return prediction uncertainties"}
    ],
    
    "hardware_requirements": ["CPU or GPU", "RAM: 2-4GB"],
    "time_complexity": "O(n) - ~1000 predictions/second on GPU",
    
    "outputs": [
        {"name": "predictions", "type": "numpy.ndarray", "description": "Predicted values shape (n_molecules, n_tasks)"},
        {"name": "uncertainties", "type": "numpy.ndarray", "description": "Prediction uncertainties (if requested)"}
    ],
    
    "failure_modes": [
        {"condition": "Invalid SMILES", "behavior": "Returns NaN for that molecule"},
        {"condition": "Featurizer mismatch", "behavior": "Raises ValueError"}
    ]
}

# =============================================================================
# COLLECTION OF ALL DEEPCHEM TOOLS
# =============================================================================

DEEPCHEM_TOOLS = [
    # Data loading
    deepchem_load_data,
    deepchem_load_molnet,
    # Featurization
    deepchem_circular_fingerprint,
    deepchem_rdkit_descriptors,
    deepchem_mol_graph_conv_featurizer,
    deepchem_featurize_molecules,
    # Data splitting
    deepchem_scaffold_splitter,
    # Models
    deepchem_gcn_model,
    deepchem_gat_model,
    deepchem_attentive_fp_model,
    deepchem_multitask_classifier,
    deepchem_multitask_regressor,
    deepchem_huggingface_model,
    # Training & Evaluation
    deepchem_train_graph_model,
    deepchem_evaluate_model,
    deepchem_predict,
]


def get_deepchem_tools_for_openai() -> list:
    """Get all DeepChem tools in OpenAI function calling format."""
    from template import get_openai_function_schema
    return [get_openai_function_schema(tool) for tool in DEEPCHEM_TOOLS]


def get_deepchem_tools_for_anthropic() -> list:
    """Get all DeepChem tools in Anthropic tool use format."""
    from template import get_anthropic_tool_schema
    return [get_anthropic_tool_schema(tool) for tool in DEEPCHEM_TOOLS]


if __name__ == "__main__":
    print(f"DeepChem Tools: {len(DEEPCHEM_TOOLS)} tools defined")
    for tool in DEEPCHEM_TOOLS:
        print(f"  - {tool['name']}")
