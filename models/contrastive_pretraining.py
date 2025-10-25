"""
Contrastive Affective Pretraining (CAP) for CAPMeme
Implements contrastive learning with affective alignment for multimodal embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class ContrastiveAffectivePretraining(nn.Module):
    """Contrastive Affective Pretraining module for embedding alignment"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        temperature: float = 0.07,
        contrastive_weight: float = 0.5,
        affective_weight: float = 0.3,
        knowledge_weight: float = 0.2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.affective_weight = affective_weight
        self.knowledge_weight = knowledge_weight
        
        # Affective alignment layers
        self.affective_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Cross-modal alignment layers
        self.image_to_text_proj = nn.Linear(embedding_dim, embedding_dim)
        self.text_to_image_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Affective-specific projections
        self.sarcasm_proj = nn.Linear(embedding_dim, embedding_dim)
        self.persuasion_proj = nn.Linear(embedding_dim, embedding_dim)
        self.harm_proj = nn.Linear(embedding_dim, embedding_dim)
        self.neutral_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def compute_contrastive_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss between image and text embeddings"""
        batch_size = image_embeddings.size(0)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        
        # Create labels for contrastive learning (diagonal should be positive)
        labels_contrastive = torch.arange(batch_size, device=image_embeddings.device)
        
        # Image-to-text contrastive loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels_contrastive)
        
        # Text-to-image contrastive loss
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels_contrastive)
        
        # Total contrastive loss
        contrastive_loss = (loss_i2t + loss_t2i) / 2
        
        return contrastive_loss
    
    def compute_affective_alignment_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        affective_labels: torch.Tensor,
        sarcasm_labels: torch.Tensor,
        persuasion_labels: torch.Tensor,
        harm_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute affective alignment loss for different affective dimensions"""
        
        # Project embeddings for affective alignment
        image_proj = self.affective_projector(image_embeddings)
        text_proj = self.affective_projector(text_embeddings)
        
        # Cross-modal projections
        image_to_text = self.image_to_text_proj(image_proj)
        text_to_image = self.text_to_image_proj(text_proj)
        
        # Align cross-modal embeddings
        aligned_image = self.layer_norm(image_embeddings + image_to_text)
        aligned_text = self.layer_norm(text_embeddings + text_to_image)
        
        # Affective-specific losses
        losses = {}
        
        # Sarcasm alignment
        sarcasm_mask = sarcasm_labels.bool()
        if sarcasm_mask.any():
            sarcasm_image_proj = self.sarcasm_proj(aligned_image[sarcasm_mask])
            sarcasm_text_proj = self.sarcasm_proj(aligned_text[sarcasm_mask])
            losses['sarcasm_loss'] = F.mse_loss(sarcasm_image_proj, sarcasm_text_proj)
        
        # Persuasion alignment
        persuasion_mask = persuasion_labels.bool()
        if persuasion_mask.any():
            persuasion_image_proj = self.persuasion_proj(aligned_image[persuasion_mask])
            persuasion_text_proj = self.persuasion_proj(aligned_text[persuasion_mask])
            losses['persuasion_loss'] = F.mse_loss(persuasion_image_proj, persuasion_text_proj)
        
        # Harm alignment
        harm_mask = harm_labels.bool()
        if harm_mask.any():
            harm_image_proj = self.harm_proj(aligned_image[harm_mask])
            harm_text_proj = self.harm_proj(aligned_text[harm_mask])
            losses['harm_loss'] = F.mse_loss(harm_image_proj, harm_text_proj)
        
        # Neutral alignment
        neutral_mask = (affective_labels == 3).bool()
        if neutral_mask.any():
            neutral_image_proj = self.neutral_proj(aligned_image[neutral_mask])
            neutral_text_proj = self.neutral_proj(aligned_text[neutral_mask])
            losses['neutral_loss'] = F.mse_loss(neutral_image_proj, neutral_text_proj)
        
        return losses
    
    def compute_knowledge_alignment_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        knowledge_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute knowledge graph alignment loss"""
        if knowledge_embeddings is None:
            return torch.tensor(0.0, device=image_embeddings.device)
        
        # Align with knowledge embeddings
        image_knowledge_sim = F.cosine_similarity(image_embeddings, knowledge_embeddings, dim=-1)
        text_knowledge_sim = F.cosine_similarity(text_embeddings, knowledge_embeddings, dim=-1)
        
        # Knowledge alignment loss (encourage similar knowledge alignment for image-text pairs)
        knowledge_loss = F.mse_loss(image_knowledge_sim, text_knowledge_sim)
        
        return knowledge_loss
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        affective_labels: torch.Tensor,
        sarcasm_labels: torch.Tensor,
        persuasion_labels: torch.Tensor,
        harm_labels: torch.Tensor,
        knowledge_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for Contrastive Affective Pretraining"""
        
        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(
            image_embeddings, text_embeddings, affective_labels
        )
        
        # Compute affective alignment losses
        affective_losses = self.compute_affective_alignment_loss(
            image_embeddings, text_embeddings, affective_labels,
            sarcasm_labels, persuasion_labels, harm_labels
        )
        
        # Compute knowledge alignment loss
        knowledge_loss = self.compute_knowledge_alignment_loss(
            image_embeddings, text_embeddings, knowledge_embeddings
        )
        
        # Combine losses
        total_affective_loss = sum(affective_losses.values()) if affective_losses else torch.tensor(0.0, device=image_embeddings.device)
        
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.affective_weight * total_affective_loss +
            self.knowledge_weight * knowledge_loss
        )
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'affective_loss': total_affective_loss,
            'knowledge_loss': knowledge_loss,
            **affective_losses
        }


class AffectiveContrastiveLoss(nn.Module):
    """Specialized contrastive loss for affective alignment"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        positive_pairs: torch.Tensor,
        negative_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Compute affective contrastive loss"""
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarities
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive and negative pairs
        positive_mask = positive_pairs.float()
        negative_mask = negative_pairs.float()
        
        # Positive loss (pull similar embeddings together)
        positive_loss = -torch.log(torch.sigmoid(similarity_matrix * positive_mask)).sum()
        
        # Negative loss (push dissimilar embeddings apart)
        negative_loss = -torch.log(torch.sigmoid(-similarity_matrix * negative_mask)).sum()
        
        # Total loss
        total_loss = positive_loss + negative_loss
        
        return total_loss


class MultiModalFusion(nn.Module):
    """Multi-modal fusion module for combining image, text, and knowledge embeddings"""
    
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Attention-based fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Affective-specific fusion
        self.affective_fusion = nn.ModuleDict({
            'sarcasm': nn.Linear(embedding_dim, embedding_dim),
            'persuasion': nn.Linear(embedding_dim, embedding_dim),
            'harm': nn.Linear(embedding_dim, embedding_dim),
            'neutral': nn.Linear(embedding_dim, embedding_dim)
        })
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        knowledge_embeddings: torch.Tensor,
        affective_labels: torch.Tensor
    ) -> torch.Tensor:
        """Fuse multimodal embeddings with affective awareness"""
        
        # Stack embeddings for attention
        stacked_embeddings = torch.stack([image_embeddings, text_embeddings, knowledge_embeddings], dim=1)
        
        # Apply attention
        attended_embeddings, _ = self.attention(
            stacked_embeddings, stacked_embeddings, stacked_embeddings
        )
        
        # Flatten and fuse
        fused_embeddings = attended_embeddings.view(attended_embeddings.size(0), -1)
        fused_output = self.fusion_layer(fused_embeddings)
        
        # Apply affective-specific processing
        affective_outputs = []
        for i, label in enumerate(affective_labels):
            affective_type = ['sarcasm', 'persuasion', 'harm', 'neutral'][label.item()]
            affective_output = self.affective_fusion[affective_type](fused_output[i:i+1])
            affective_outputs.append(affective_output)
        
        affective_fused = torch.cat(affective_outputs, dim=0)
        
        return affective_fused
