"""
Inference Script for CAPMeme
Load trained model and make predictions on new meme data
"""

import torch
import argparse
import json
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import yaml

from models.capmeme import CAPMeme
from utils.config import load_config


def load_model(checkpoint_path: str, config_path: str):
    """Load trained CAPMeme model"""
    config = load_config(config_path)
    model = CAPMeme(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model, config


def preprocess_image(image_path: str, image_size: int = 224):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def preprocess_text(text: str, tokenizer, max_length: int = 128):
    """Preprocess text for inference"""
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return encoding['input_ids'], encoding['attention_mask']


def predict_meme(model, image_path: str, text: str, config: dict):
    """Make prediction on a single meme"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load tokenizer
    tokenizer_name = config['model']['text_encoder']
    if tokenizer_name == 'bert':
        tokenizer_name = 'bert-base-uncased'
    elif tokenizer_name == 'llama':
        tokenizer_name = 'meta-llama/Llama-2-7b-hf'
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Preprocess inputs
    image_tensor = preprocess_image(image_path, config['data']['image_size'])
    input_ids, attention_mask = preprocess_text(text, tokenizer, config['data']['max_text_length'])
    
    # Move to device
    image_tensor = image_tensor.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            images=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=[text]
        )
    
    predictions = outputs['predictions']
    
    # Get predicted class
    class_names = ['sarcastic', 'persuasive', 'harmful', 'neutral']
    predicted_class = torch.argmax(predictions['affective'], dim=-1).item()
    confidence = torch.max(torch.softmax(predictions['affective'], dim=-1)).item()
    
    # Get individual task predictions
    sarcasm_prob = torch.softmax(predictions['sarcasm'], dim=-1)[0, 1].item()
    persuasion_prob = torch.softmax(predictions['persuasion'], dim=-1)[0, 1].item()
    harm_prob = torch.softmax(predictions['harm'], dim=-1)[0, 1].item()
    
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence,
        'sarcasm_probability': sarcasm_prob,
        'persuasion_probability': persuasion_prob,
        'harm_probability': harm_prob,
        'class_probabilities': {
            class_names[i]: torch.softmax(predictions['affective'], dim=-1)[0, i].item()
            for i in range(len(class_names))
        }
    }


def main():
    parser = argparse.ArgumentParser(description='CAPMeme Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to meme image')
    parser.add_argument('--text', type=str, required=True,
                       help='Meme text')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading CAPMeme model...")
    model, config = load_model(args.checkpoint, args.config)
    
    # Make prediction
    print("Making prediction...")
    result = predict_meme(model, args.image, args.text, config)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Sarcasm Probability: {result['sarcasm_probability']:.4f}")
    print(f"Persuasion Probability: {result['persuasion_probability']:.4f}")
    print(f"Harm Probability: {result['harm_probability']:.4f}")
    print("\nClass Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
