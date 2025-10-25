"""
Dataset loading and preprocessing for CAPMeme
Handles multimodal meme data with image, text, and affective labels
"""

import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import numpy as np


class MemeDataset(Dataset):
    """Multimodal meme dataset with image, text, and affective labels"""
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer_name: str = "bert-base-uncased",
        image_size: int = 224,
        max_text_length: int = 128,
        split: str = "train"
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.data = self._load_data()
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Label mapping
        self.label_map = {
            'sarcastic': 0,
            'persuasive': 1, 
            'harmful': 2,
            'neutral': 3
        }
    
    def _load_data(self) -> List[Dict]:
        """Load dataset from file"""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Create sample data if file doesn't exist
            data = self._create_sample_data()
        
        return data
    
    def _create_sample_data(self) -> List[Dict]:
        """Create large-scale dataset for 10,000 images"""
        import random
        
        # Sample texts for different categories
        sarcastic_texts = [
            "When you realize it's Monday again", "Oh great, another meeting", 
            "Sure, let's have another zoom call", "Perfect timing as always",
            "This is exactly what I needed today", "Oh wonderful, more work",
            "Just what I wanted to hear", "Fantastic, another deadline",
            "Oh joy, another presentation", "This will be fun, I'm sure"
        ]
        
        persuasive_texts = [
            "Buy this amazing product now!", "Limited time offer - act fast!",
            "You won't believe this deal!", "Transform your life today!",
            "Don't miss out on this opportunity!", "Revolutionary new product!",
            "Join thousands of satisfied customers!", "Get yours before it's gone!",
            "This will change everything!", "Exclusive offer just for you!"
        ]
        
        harmful_texts = [
            "You're worthless and should give up", "Nobody likes you anyway",
            "You'll never amount to anything", "Just quit while you're ahead",
            "You're a complete failure", "No one cares about you",
            "You're not good enough", "Stop trying, you'll never succeed",
            "You're embarrassing yourself", "Just disappear already"
        ]
        
        neutral_texts = [
            "Nice weather today", "The sky is blue", "Coffee tastes good",
            "It's a beautiful day", "The sun is shining", "Birds are singing",
            "The grass is green", "Water is wet", "Fire is hot", "Ice is cold"
        ]
        
        # Create 10,000 samples
        sample_data = []
        for i in range(10000):
            # Determine category (balanced distribution)
            category_idx = i % 4
            categories = ['sarcastic', 'persuasive', 'harmful', 'neutral']
            category = categories[category_idx]
            
            # Select appropriate text
            if category == 'sarcastic':
                text = random.choice(sarcastic_texts)
                sarcasm_label, persuasion_label, harm_label = 1, 0, 0
            elif category == 'persuasive':
                text = random.choice(persuasive_texts)
                sarcasm_label, persuasion_label, harm_label = 0, 1, 0
            elif category == 'harmful':
                text = random.choice(harmful_texts)
                sarcasm_label, persuasion_label, harm_label = 0, 0, 1
            else:  # neutral
                text = random.choice(neutral_texts)
                sarcasm_label, persuasion_label, harm_label = 0, 0, 0
            
            sample_data.append({
                "id": f"meme_{i+1:05d}",
                "image_path": f"images/meme_{i+1:05d}.jpg",
                "text": text,
                "sarcasm_label": sarcasm_label,
                "persuasion_label": persuasion_label,
                "harm_label": harm_label,
                "affective_label": category
            })
        
        # Save sample data
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"Created dataset with {len(sample_data)} samples")
        return sample_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset"""
        item = self.data[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, item["image_path"])
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transforms(image)
        except:
            # Create dummy image if file doesn't exist
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Process text
        text = item["text"]
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels
        affective_label = self.label_map.get(item["affective_label"], 3)
        
        return {
            'image': image.squeeze(0),
            'text': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'affective_label': torch.tensor(affective_label, dtype=torch.long),
            'sarcasm_label': torch.tensor(item["sarcasm_label"], dtype=torch.long),
            'persuasion_label': torch.tensor(item["persuasion_label"], dtype=torch.long),
            'harm_label': torch.tensor(item["harm_label"], dtype=torch.long),
            'id': item["id"]
        }


def create_data_loaders(
    config: Dict,
    data_path: str,
    image_dir: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders for 10K dataset"""
    
    # Load full dataset first
    full_dataset = MemeDataset(
        data_path=data_path,
        image_dir=image_dir,
        image_size=config['data']['image_size'],
        max_text_length=config['data']['max_text_length'],
        split='full'
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create splits
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders with optimized settings for GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches for faster loading
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'text': torch.stack([item['text'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'affective_label': torch.stack([item['affective_label'] for item in batch]),
        'sarcasm_label': torch.stack([item['sarcasm_label'] for item in batch]),
        'persuasion_label': torch.stack([item['persuasion_label'] for item in batch]),
        'harm_label': torch.stack([item['harm_label'] for item in batch]),
        'id': [item['id'] for item in batch]
    }
