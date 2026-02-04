"""
DeepChem - Train Graph Convolutional Model Tool Description
"""

tool_description = {
    "name": "deepchem_train_graph_model",
    "description": (
        "Train a Graph Convolutional Network (GCN) for molecular property prediction using DeepChem. "
        "Implements message passing on molecular graphs with configurable architecture. "
        "State-of-the-art performance on MoleculeNet benchmarks; outperforms traditional ML on large datasets (>1K samples). "
        "Supports multi-task learning, classification and regression, with automatic GPU acceleration. "
        "Best choice for datasets >5K molecules with rich structural diversity."
    ),
    "required_parameters": [
        {
            "name": "dataset",
            "type": "dc.data.Dataset",
            "default": None,
            "description": (
                "DeepChem dataset object containing featurized molecular graphs. "
                "Must use MolGraphConvFeaturizer for featurization. "
                "Dataset should have X (features), y (labels), and w (weights) attributes. "
                "Typical creation: dc.data.NumpyDataset(X=graphs, y=labels, ids=smiles)."
            ),
        },
        {
            "name": "n_tasks",
            "type": "int",
            "default": None,
            "description": (
                "Number of prediction tasks (output dimensions). "
                "Single-task: 1. Multi-task: number of properties to predict simultaneously. "
                "Example: Tox21 has 12 tasks (12 different toxicity endpoints). "
                "Must match dataset.y.shape[1]."
            ),
        },
    ],
    "optional_parameters": [
        {
            "name": "graph_conv_layers",
            "type": "list[int]",
            "default": [64, 64],
            "description": (
                "Sizes of graph convolution layers. Each int is hidden dimension for that layer. "
                "Default [64, 64] = two GC layers with 64 units each. "
                "Deeper/wider networks (e.g., [128, 128, 128]) improve capacity but risk overfitting on small data. "
                "Performance impact: ~linear in number of layers and quadratic in layer size."
            ),
        },
        {
            "name": "dense_layer_size",
            "type": "int",
            "default": 128,
            "description": (
                "Size of dense (fully-connected) layers after graph convolutions. "
                "Typical range: 64-512. "
                "Larger values increase model capacity and training time. "
                "Minimal impact on inference latency compared to graph convolution layers."
            ),
        },
        {
            "name": "dropout",
            "type": "float",
            "default": 0.0,
            "description": (
                "Dropout probability for regularization (range 0.0-1.0). "
                "Recommended: 0.1-0.3 for small datasets (<5K), 0.0 for large datasets (>50K). "
                "Higher dropout prevents overfitting but may slow convergence. "
                "Applied to dense layers only, not graph convolution layers."
            ),
        },
        {
            "name": "mode",
            "type": "str",
            "default": "classification",
            "description": (
                "Task type: 'classification' or 'regression'. "
                "Classification: binary or multi-class prediction with sigmoid/softmax output. "
                "Regression: continuous value prediction with linear output. "
                "Affects loss function and final layer activation."
            ),
        },
        {
            "name": "batch_size",
            "type": "int",
            "default": 50,
            "description": (
                "Number of samples per training batch. "
                "Typical range: 32-128. Larger batches use more GPU memory but may converge faster. "
                "GPU memory usage scales linearly with batch_size. "
                "Reduce if encountering OOM errors; increase for faster training on large datasets."
            ),
        },
        {
            "name": "nb_epoch",
            "type": "int",
            "default": 50,
            "description": (
                "Number of training epochs (full passes through dataset). "
                "Typical range: 20-100. More epochs improve convergence but risk overfitting. "
                "Use early stopping or validation monitoring to determine optimal value. "
                "Training time scales linearly with nb_epoch."
            ),
        },
        {
            "name": "learning_rate",
            "type": "float",
            "default": 0.001,
            "description": (
                "Optimizer learning rate. Typical range: 0.0001-0.01. "
                "Default 0.001 works well for most cases. "
                "Lower LR: more stable but slower convergence. Higher LR: faster but may diverge. "
                "Consider learning rate scheduling for large datasets."
            ),
        },
        {
            "name": "use_queue",
            "type": "bool",
            "default": True,
            "description": (
                "Use TensorFlow queue-based data loading for better GPU utilization. "
                "Recommended: True for datasets >10K molecules. "
                "Improves training speed by ~20-30% but uses more CPU. "
                "Set False if encountering queue-related errors or on CPU-only systems."
            ),
        },
    ],
    "hardware_requirements": {
        "device": "gpu_optional",
        "notes": (
            "CPU-only: Functional but slow (10-50x slower than GPU for large datasets). "
            "GPU: Highly recommended for datasets >5K molecules. "
            "GPU memory requirements: "
            "  - Small datasets (<10K): 2GB VRAM sufficient "
            "  - Medium datasets (10K-100K): 4-8GB VRAM "
            "  - Large datasets (>100K): 8-16GB VRAM "
            "Training on CPU: feasible for <1K molecules. "
            "System RAM: 8GB minimum, 16GB+ recommended. "
            "Compatible GPUs: NVIDIA with CUDA support (Tesla, RTX, A100)."
        ),
    },
    "time_complexity": {
        "assumptions": (
            "Wall-clock training time for specified epochs on representative hardware. "
            "Dataset: 1K molecules, average 30 atoms, single task. "
            "CPU: Apple M1 (8 cores). GPU: NVIDIA RTX 3080 (10GB VRAM). "
            "Includes forward pass, backpropagation, and validation. "
            "Times scale linearly with nb_epoch and dataset size. "
            "Graph size (atoms/bonds) has sublinear impact due to batching."
        ),
        "latency_seconds": {
            "1k_cpu_10_epochs": 180,  # 1K molecules, 10 epochs, CPU
            "1k_gpu_10_epochs": 15,   # 1K molecules, 10 epochs, GPU
            "10k_gpu_50_epochs": 450, # 10K molecules, 50 epochs, GPU
            "100k_gpu_50_epochs": 3600, # 100K molecules, 50 epochs, GPU  
            "single_epoch_1k_gpu": 1.5,  # Per-epoch time
        },
    },
    "outputs": {
        "type": "dc.models.GraphConvModel",
        "schema": {
            "model_type": "trained DeepChem model object",
            "methods": ["predict()", "evaluate()", "save_checkpoint()"],
            "attributes": ["model_dir", "n_tasks", "mode"],
        },
        "example": {
            "usage": "predictions = model.predict(test_dataset)",
            "prediction_shape": "(n_molecules, n_tasks)",
            "classification_output": "probability scores [0, 1]",
            "regression_output": "continuous values",
        },
    },
    "failure_modes": [
        {
            "error": "CUDA out of memory",
            "cause": "Batch size too large for available GPU memory",
            "fix": "Reduce batch_size (try 32, 16, 8), use smaller model (fewer/narrower layers), or use CPU",
        },
        {
            "error": "TypeError: dataset must be featurized with MolGraphConvFeaturizer",
            "cause": "Dataset was featurized with wrong featurizer (e.g., CircularFingerprint)",
            "fix": "Re-featurize dataset using dc.feat.MolGraphConvFeaturizer()",
        },
        {
            "error": "NaN loss during training",
            "cause": "Learning rate too high, gradient explosion, or invalid inputs",
            "fix": "Reduce learning_rate (try 0.0001), add gradient clipping, check for invalid molecules in dataset",
        },
        {
            "error": "Model not improving (loss plateaus)",
            "cause": "Learning rate too low, insufficient model capacity, or data quality issues",
            "fix": "Increase learning_rate, use larger/deeper model, check dataset quality and labels",
        },
        {
            "error": "Slow training on GPU",
            "cause": "Small batch size underutilizing GPU, or data loading bottleneck",
            "fix": "Increase batch_size (64-128), ensure use_queue=True, verify GPU utilization with nvidia-smi",
        },
    ],
}
