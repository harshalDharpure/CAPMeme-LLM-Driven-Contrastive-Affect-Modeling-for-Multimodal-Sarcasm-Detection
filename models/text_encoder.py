"""
Text Encoder for CAPMeme
Supports BERT and LLaMA-based models for deeper textual context understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, BertTokenizer, BertConfig,
    LlamaModel, LlamaTokenizer, LlamaConfig
)
from typing import Dict, Tuple, Optional, Union
import math


class BERTTextEncoder(nn.Module):
    """BERT-based text encoder for deeper contextual understanding"""
    
    def __init__(self, model_name: str = "bert-base-uncased", embedding_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load BERT model
        self.bert_model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Projection layer to match desired embedding dimension
        self.projection = nn.Linear(
            self.bert_model.config.hidden_size,
            embedding_dim
        )
        
        # Additional layers for enhanced text understanding
        self.context_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for text encoding"""
        # Get BERT outputs
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Project to desired dimension
        projected_embeddings = self.projection(cls_embeddings)
        
        # Apply self-attention for enhanced context
        attended_embeddings, _ = self.context_attention(
            projected_embeddings.unsqueeze(1),
            projected_embeddings.unsqueeze(1),
            projected_embeddings.unsqueeze(1)
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(projected_embeddings + attended_embeddings.squeeze(1))
        output = self.dropout(output)
        
        # Normalize embeddings
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class LLaMATextEncoder(nn.Module):
    """LLaMA-based text encoder for advanced language understanding"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", embedding_dim: int = 512):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load LLaMA model
        self.llama_model = LlamaModel.from_pretrained(model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        
        # Projection layer
        self.projection = nn.Linear(
            self.llama_model.config.hidden_size,
            embedding_dim
        )
        
        # Enhanced context processing
        self.context_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for text encoding"""
        # Get LLaMA outputs
        llama_outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use last token representation (or mean pooling)
        if attention_mask is not None:
            # Mean pooling over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(llama_outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(llama_outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_embeddings = sum_embeddings / sum_mask
        else:
            # Use last token
            pooled_embeddings = llama_outputs.last_hidden_state[:, -1, :]
        
        # Project to desired dimension
        projected_embeddings = self.projection(pooled_embeddings)
        
        # Enhanced context processing
        enhanced_embeddings = self.context_processor(projected_embeddings)
        enhanced_embeddings = self.dropout(enhanced_embeddings)
        
        # Normalize embeddings
        enhanced_embeddings = F.normalize(enhanced_embeddings, p=2, dim=-1)
        
        return enhanced_embeddings


class TextEncoder(nn.Module):
    """Main text encoder supporting BERT and LLaMA"""
    
    def __init__(self, encoder_type: str = "bert", model_name: Optional[str] = None,
                 embedding_dim: int = 512):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.embedding_dim = embedding_dim
        
        if self.encoder_type == "bert":
            model_name = model_name or "bert-base-uncased"
            self.encoder = BERTTextEncoder(model_name, embedding_dim)
        elif self.encoder_type == "llama":
            model_name = model_name or "meta-llama/Llama-2-7b-hf"
            self.encoder = LLaMATextEncoder(model_name, embedding_dim)
        else:
            raise ValueError(f"Unsupported text encoder type: {encoder_type}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.encoder(input_ids, attention_mask)
    
    def get_tokenizer(self):
        """Get the tokenizer for this encoder"""
        return self.encoder.tokenizer


class AffectiveTextProcessor(nn.Module):
    """Specialized processor for affective text understanding"""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Affective-specific layers
        self.sarcasm_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.persuasion_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.harm_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process text embeddings for different affective dimensions"""
        # Process for different affective dimensions
        sarcasm_features = self.sarcasm_processor(text_embeddings)
        persuasion_features = self.persuasion_processor(text_embeddings)
        harm_features = self.harm_processor(text_embeddings)
        
        # Fuse all features
        fused_features = torch.cat([sarcasm_features, persuasion_features, harm_features], dim=-1)
        fused_output = self.fusion_layer(fused_features)
        
        return {
            'sarcasm_features': sarcasm_features,
            'persuasion_features': persuasion_features,
            'harm_features': harm_features,
            'fused_features': fused_output
        }
