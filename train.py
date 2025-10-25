"""
Main Training Script for CAPMeme
Implements the complete training pipeline with evaluation and logging
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

# Import model components
from models.capmeme import CAPMeme, CAPMemeTrainer
from data.dataset import create_data_loaders
from evaluation.metrics import CAPMemeEvaluator
from utils.config import load_config
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CAPMeme model')
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
    parser.add_argument('--wandb_project', type=str, default='capmeme',
                       help='Weights & Biases project name')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def setup_device(device_arg: str):
    """Setup device for training"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def setup_wandb(config: dict, project_name: str):
    """Setup Weights & Biases logging"""
    if config['logging']['use_wandb']:
        wandb.init(
            project=project_name,
            config=config,
            name=f"capmeme_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )


def create_sample_data(data_path: str, image_dir: str):
    """Create sample data for demonstration"""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Create sample dataset
    sample_data = [
        {
            "id": "sample_1",
            "image_path": "meme1.jpg",
            "text": "When you realize it's Monday again",
            "sarcasm_label": 1,
            "persuasion_label": 0,
            "harm_label": 0,
            "affective_label": "sarcastic"
        },
        {
            "id": "sample_2",
            "image_path": "meme2.jpg", 
            "text": "Buy this amazing product now!",
            "sarcasm_label": 0,
            "persuasion_label": 1,
            "harm_label": 0,
            "affective_label": "persuasive"
        },
        {
            "id": "sample_3",
            "image_path": "meme3.jpg",
            "text": "You're worthless and should give up",
            "sarcasm_label": 0,
            "persuasion_label": 0,
            "harm_label": 1,
            "affective_label": "harmful"
        },
        {
            "id": "sample_4",
            "image_path": "meme4.jpg",
            "text": "Nice weather today",
            "sarcasm_label": 0,
            "persuasion_label": 0,
            "harm_label": 0,
            "affective_label": "neutral"
        }
    ]
    
    # Save dataset
    with open(data_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample dataset at {data_path}")


def train_model(args):
    """Main training function"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(args.device)
    
    # Setup logging
    logger = setup_logger(args.output_dir)
    logger.info("Starting CAPMeme training")
    logger.info(f"Configuration: {config}")
    
    # Setup Weights & Biases
    setup_wandb(config, args.wandb_project)
    
    # Create sample data if it doesn't exist
    if not os.path.exists(args.data_path):
        logger.info("Creating sample dataset")
        create_sample_data(args.data_path, args.image_dir)
    
    # Create data loaders
    logger.info("Loading dataset")
    train_loader, val_loader, test_loader = create_data_loaders(
        config, args.data_path, args.image_dir
    )
    
    # Initialize model
    logger.info("Initializing CAPMeme model")
    model = CAPMeme(config)
    model.to(device)
    
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
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, evaluator)
        
        # Validate
        val_loss, val_metrics = trainer.validate(val_loader, evaluator)
        
        # Update learning rate
        trainer.scheduler.step()
        
        # Log metrics
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
                'learning_rate': trainer.scheduler.get_last_lr()[0]
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
            'test_auroc_macro': test_metrics['affective_auroc_macro']
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
