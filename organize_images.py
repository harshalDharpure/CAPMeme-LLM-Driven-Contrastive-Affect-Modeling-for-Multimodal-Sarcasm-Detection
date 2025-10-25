"""
Image Organization Script for CAPMeme
Helps organize 10,000 images into the required directory structure
"""

import os
import shutil
import argparse
from pathlib import Path
import json
from typing import List, Dict
import random


def organize_images(source_dir: str, target_dir: str, dataset_size: int = 10000):
    """
    Organize images from source directory to target directory
    
    Args:
        source_dir: Directory containing your 10,000 images
        target_dir: Target directory for organized images
        dataset_size: Number of images to process
    """
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files from source directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    source_path = Path(source_dir)
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(source_path.glob(f'*{ext}'))
        image_files.extend(source_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} image files in {source_dir}")
    
    if len(image_files) < dataset_size:
        print(f"Warning: Only {len(image_files)} images found, but {dataset_size} requested")
        dataset_size = len(image_files)
    
    # Shuffle and select images
    random.shuffle(image_files)
    selected_images = image_files[:dataset_size]
    
    # Copy images with new naming convention
    print(f"Copying {len(selected_images)} images...")
    
    for i, image_path in enumerate(selected_images):
        new_name = f"meme_{i+1:05d}.jpg"
        target_path = os.path.join(target_dir, new_name)
        
        try:
            shutil.copy2(image_path, target_path)
            if (i + 1) % 1000 == 0:
                print(f"Copied {i + 1} images...")
        except Exception as e:
            print(f"Error copying {image_path}: {e}")
    
    print(f"Successfully organized {len(selected_images)} images to {target_dir}")
    return len(selected_images)


def create_dataset_json(image_dir: str, output_path: str, num_images: int = 10000):
    """
    Create dataset JSON file for the organized images
    
    Args:
        image_dir: Directory containing organized images
        output_path: Path to save the dataset JSON
        num_images: Number of images to include in dataset
    """
    
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
    
    # Create dataset entries
    dataset = []
    for i in range(num_images):
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
        
        dataset.append({
            "id": f"meme_{i+1:05d}",
            "image_path": f"images/meme_{i+1:05d}.jpg",
            "text": text,
            "sarcasm_label": sarcasm_label,
            "persuasion_label": persuasion_label,
            "harm_label": harm_label,
            "affective_label": category
        })
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created dataset JSON with {len(dataset)} entries at {output_path}")
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description='Organize images for CAPMeme training')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Directory containing your 10,000 images')
    parser.add_argument('--target_dir', type=str, default='data/images',
                       help='Target directory for organized images')
    parser.add_argument('--dataset_json', type=str, default='data/meme_dataset.json',
                       help='Path to save dataset JSON file')
    parser.add_argument('--num_images', type=int, default=10000,
                       help='Number of images to process')
    parser.add_argument('--skip_copy', action='store_true',
                       help='Skip copying images, only create JSON')
    
    args = parser.parse_args()
    
    print("CAPMeme Image Organization Script")
    print("=" * 50)
    
    # Check if source directory exists
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist")
        return
    
    # Organize images
    if not args.skip_copy:
        print(f"Organizing images from {args.source_dir} to {args.target_dir}")
        num_copied = organize_images(args.source_dir, args.target_dir, args.num_images)
    else:
        print("Skipping image copying (--skip_copy flag set)")
        num_copied = args.num_images
    
    # Create dataset JSON
    print(f"Creating dataset JSON file...")
    create_dataset_json(args.target_dir, args.dataset_json, num_copied)
    
    print("\n" + "=" * 50)
    print("Image organization completed!")
    print(f"Images organized: {num_copied}")
    print(f"Target directory: {args.target_dir}")
    print(f"Dataset JSON: {args.dataset_json}")
    print("\nNext steps:")
    print("1. Verify your images are in the correct directory")
    print("2. Run: python train_gpu.py --data_path data/meme_dataset.json --image_dir data/images")
    print("3. Monitor training progress with Weights & Biases")


if __name__ == "__main__":
    main()
