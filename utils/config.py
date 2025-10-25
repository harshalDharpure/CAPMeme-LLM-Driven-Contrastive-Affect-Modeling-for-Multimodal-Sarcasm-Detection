"""
Utility functions for CAPMeme
Configuration loading, logging, and other helper functions
"""

import yaml
import os
import logging
from typing import Dict, Any
import torch
import numpy as np
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logger(output_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """Setup logger for training"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('capmeme')
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return f"{size_all_mb:.2f} MB"


def save_predictions(predictions: torch.Tensor, labels: torch.Tensor, 
                    ids: list, save_path: str):
    """Save predictions to file"""
    import pandas as pd
    
    pred_labels = torch.argmax(predictions, dim=-1)
    
    df = pd.DataFrame({
        'id': ids,
        'predicted_label': pred_labels.cpu().numpy(),
        'true_label': labels.cpu().numpy(),
        'confidence': torch.max(torch.softmax(predictions, dim=-1), dim=-1)[0].cpu().numpy()
    })
    
    df.to_csv(save_path, index=False)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, save_path: str, **kwargs):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, save_path)


def create_directory_structure(base_dir: str):
    """Create standard directory structure for the project"""
    directories = [
        'checkpoints',
        'logs',
        'results',
        'plots',
        'data/raw',
        'data/processed',
        'data/conceptnet_cache',
        'models/saved',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
    }
    
    if torch.cuda.is_available():
        device_info['cuda_version'] = torch.version.cuda
        device_info['cudnn_version'] = torch.backends.cudnn.version()
    
    return device_info


def print_model_summary(model: torch.nn.Module):
    """Print model summary"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size(model)}")
    print(f"Device: {next(model.parameters()).device}")
    
    # Print layer information
    print("\nModel Architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                print(f"  {name}: {module.__class__.__name__} ({param_count:,} parameters)")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration file"""
    required_keys = [
        'model', 'data', 'training', 'cap', 'knowledge_graph', 
        'evaluation', 'logging'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required configuration key: {key}")
            return False
    
    # Validate model config
    model_config = config['model']
    required_model_keys = ['vision_encoder', 'text_encoder', 'embedding_dim', 'num_classes']
    for key in required_model_keys:
        if key not in model_config:
            print(f"Missing required model configuration key: {key}")
            return False
    
    return True


def setup_data_paths(config: Dict[str, Any], base_dir: str = '.') -> Dict[str, str]:
    """Setup data paths based on configuration"""
    paths = {
        'data_dir': os.path.join(base_dir, 'data'),
        'image_dir': os.path.join(base_dir, 'data', 'images'),
        'dataset_file': os.path.join(base_dir, 'data', 'meme_dataset.json'),
        'cache_dir': os.path.join(base_dir, 'data', 'conceptnet_cache'),
        'output_dir': os.path.join(base_dir, 'outputs'),
        'checkpoint_dir': os.path.join(base_dir, 'outputs', 'checkpoints'),
        'log_dir': os.path.join(base_dir, 'outputs', 'logs'),
        'result_dir': os.path.join(base_dir, 'outputs', 'results')
    }
    
    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths
