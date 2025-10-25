"""
GPU-Optimized Training Script for CAPMeme with 10K Images
Optimized for large-scale training with memory efficiency and performance
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import time

# Import model components
from models.capmeme import CAPMeme, CAPMemeTrainer
from data.dataset import create_data_loaders
from evaluation.metrics import CAPMemeEvaluator
from utils.config import load_config, setup_device, set_seed
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CAPMeme model with 10K images')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default='data/meme_dataset.json',
                       help='Path to dataset file')
    parser.add_argument('--image_dir', type=str, default='data/images',
                       help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--wandb_project', type=str, default='capmeme-10k',
                       help='Weights & Biases project name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    
    return parser.parse_args()


def setup_gpu_environment():
    """Setup GPU environment for optimal performance"""
    if torch.cuda.is_available():
        # Set CUDA memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Enable cuDNN deterministic for reproducibility (slower but consistent)
        # torch.backends.cudnn.deterministic = True
        
        print(f"GPU Setup:")
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        print(f"  - GPU Count: {torch.cuda.device_count()}")
        print(f"  - Current GPU: {torch.cuda.current_device()}")
        print(f"  - GPU Name: {torch.cuda.get_device_name()}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  - cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return True
    else:
        print("CUDA not available. Training will be slower on CPU.")
        return False


def create_sample_data(data_path: str, image_dir: str):
    """Create sample data for demonstration with 10K images"""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    print(f"Creating dataset with 10,000 samples...")
    print(f"Data will be saved to: {data_path}")
    print(f"Images directory: {image_dir}")
    print("Note: You need to place your actual images in the images directory")
    
    # The dataset creation will be handled by the MemeDataset class
    # This just ensures directories exist


def train_model(args):
    """Main training function optimized for GPU"""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup GPU environment
    gpu_available = setup_gpu_environment()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    logger = setup_logger(args.output_dir)
    logger.info("Starting CAPMeme training with 10K images")
    logger.info(f"Configuration: {config}")
    logger.info(f"GPU Available: {gpu_available}")
    
    # Setup Weights & Biases
    if config['logging']['use_wandb']:
        wandb.init(
            project=args.wandb_project,
            config=config,
            name=f"capmeme_10k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create sample data if it doesn't exist
    if not os.path.exists(args.data_path):
        logger.info("Creating sample dataset structure")
        create_sample_data(args.data_path, args.image_dir)
    
    # Create data loaders
    logger.info("Loading dataset with 10K images")
    start_time = time.time()
    train_loader, val_loader, test_loader = create_data_loaders(
        config, args.data_path, args.image_dir
    )
    load_time = time.time() - start_time
    logger.info(f"Dataset loaded in {load_time:.2f} seconds")
    
    # Initialize model
    logger.info("Initializing CAPMeme model")
    model = CAPMeme(config)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize trainer
    trainer = CAPMemeTrainer(model, config)
    
    # Initialize evaluator
    evaluator = CAPMemeEvaluator()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    logger.info("Starting training")
    best_val_accuracy = 0.0
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = time.time()
        
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, evaluator)
        
        # Validate
        val_loss, val_metrics = trainer.validate(val_loader, evaluator)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['affective_accuracy']:.4f}")
        logger.info(f"Val F1 Macro: {val_metrics['affective_f1_macro']:.4f}")
        logger.info(f"Val AUROC Macro: {val_metrics['affective_auroc_macro']:.4f}")
        
        # Log to wandb
        if config['logging']['use_wandb']:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_metrics['affective_accuracy'],
                'val_f1_macro': val_metrics['affective_f1_macro'],
                'val_auroc_macro': val_metrics['affective_auroc_macro'],
                'learning_rate': trainer.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            })
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            model.save_model(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_metrics['affective_accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['affective_accuracy']
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            model.save_model(best_model_path)
            logger.info(f"New best model saved: {best_model_path}")
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and epoch % 10 == 0:
            torch.cuda.empty_cache()
    
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time/3600:.2f} hours")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set")
    test_loss, test_metrics = trainer.validate(test_loader, evaluator)
    
    logger.info("Test Results:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['affective_accuracy']:.4f}")
    logger.info(f"Test F1 Macro: {test_metrics['affective_f1_macro']:.4f}")
    logger.info(f"Test AUROC Macro: {test_metrics['affective_auroc_macro']:.4f}")
    
    # Log final test results to wandb
    if config['logging']['use_wandb']:
        wandb.log({
            'test_loss': test_loss,
            'test_accuracy': test_metrics['affective_accuracy'],
            'test_f1_macro': test_metrics['affective_f1_macro'],
            'test_auroc_macro': test_metrics['affective_auroc_macro'],
            'total_training_time': total_training_time
        })
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    model.save_model(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
    
    # Generate evaluation plots
    evaluator.plot_training_curves(os.path.join(args.output_dir, 'training_curves.png'))
    
    logger.info("Training completed successfully!")
    
    # Finish wandb run
    if config['logging']['use_wandb']:
        wandb.finish()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()
