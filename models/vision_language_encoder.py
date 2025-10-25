"""
Vision-Language Encoder for CAPMeme
Supports CLIP and BLIP models for multimodal embedding extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, BlipModel, BlipProcessor
from typing import Dict, Tuple, Optional
import clip


class CLIPEncoder(nn.Module):
    """CLIP-based vision-language encoder"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", embedding_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Projection layers to match desired embedding dimension
        self.image_projection = nn.Linear(
            self.clip_model.config.vision_config.hidden_size, 
            embedding_dim
        )
        self.text_projection = nn.Linear(
            self.clip_model.config.text_config.hidden_size,
            embedding_dim
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP vision encoder"""
        # Get image features from CLIP
        image_features = self.clip_model.get_image_features(pixel_values=images)
        
        # Project to desired dimension
        image_embeddings = self.image_projection(image_features)
        image_embeddings = self.dropout(image_embeddings)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        
        return image_embeddings
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text using CLIP text encoder"""
        # Get text features from CLIP
        text_features = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Project to desired dimension
        text_embeddings = self.text_projection(text_features)
        text_embeddings = self.dropout(text_embeddings)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings
    
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both image and text encoding"""
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        return image_embeddings, text_embeddings


class BLIPEncoder(nn.Module):
    """BLIP-based vision-language encoder"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", embedding_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load BLIP model
        self.blip_model = BlipModel.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        
        # Projection layers
        self.image_projection = nn.Linear(
            self.blip_model.config.vision_config.hidden_size,
            embedding_dim
        )
        self.text_projection = nn.Linear(
            self.blip_model.config.text_config.hidden_size,
            embedding_dim
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using BLIP vision encoder"""
        # Get image features
        vision_outputs = self.blip_model.vision_model(pixel_values=images)
        image_features = vision_outputs.pooler_output
        
        # Project to desired dimension
        image_embeddings = self.image_projection(image_features)
        image_embeddings = self.dropout(image_embeddings)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        
        return image_embeddings
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text using BLIP text encoder"""
        # Get text features
        text_outputs = self.blip_model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.pooler_output
        
        # Project to desired dimension
        text_embeddings = self.text_projection(text_features)
        text_embeddings = self.dropout(text_embeddings)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings
    
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both image and text encoding"""
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        return image_embeddings, text_embeddings


class VisionLanguageEncoder(nn.Module):
    """Main vision-language encoder that supports both CLIP and BLIP"""
    
    def __init__(self, encoder_type: str = "clip", model_name: Optional[str] = None, 
                 embedding_dim: int = 512):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.embedding_dim = embedding_dim
        
        if self.encoder_type == "clip":
            model_name = model_name or "openai/clip-vit-base-patch32"
            self.encoder = CLIPEncoder(model_name, embedding_dim)
        elif self.encoder_type == "blip":
            model_name = model_name or "Salesforce/blip-image-captioning-base"
            self.encoder = BLIPEncoder(model_name, embedding_dim)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        return self.encoder(images, input_ids, attention_mask)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images only"""
        return self.encoder.encode_image(images)
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text only"""
        return self.encoder.encode_text(input_ids, attention_mask)
    
    def get_similarity(self, image_embeddings: torch.Tensor, 
                      text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity between image and text embeddings"""
        # Compute cosine similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.T)
        return similarity
