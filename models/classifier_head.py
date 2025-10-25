"""
Classifier Head for CAPMeme
Final classification layer for meme label prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MultiTaskClassifierHead(nn.Module):
    """Multi-task classifier head for different affective dimensions"""
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 4,
        num_tasks: int = 4  # sarcasm, persuasion, harm, overall affective
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        
        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            'sarcasm': nn.Linear(hidden_dim, 2),  # binary: sarcastic/not sarcastic
            'persuasion': nn.Linear(hidden_dim, 2),  # binary: persuasive/not persuasive
            'harm': nn.Linear(hidden_dim, 2),  # binary: harmful/not harmful
            'affective': nn.Linear(hidden_dim, num_classes)  # multi-class: overall affective label
        })
        
        # Attention mechanism for task-specific features
        self.task_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task classification"""
        
        # Process shared features
        shared_features = self.shared_layers(embeddings)
        
        # Apply task-specific attention
        attended_features, attention_weights = self.task_attention(
            shared_features.unsqueeze(1),
            shared_features.unsqueeze(1),
            shared_features.unsqueeze(1)
        )
        
        # Residual connection and layer norm
        enhanced_features = self.layer_norm(shared_features + attended_features.squeeze(1))
        
        # Generate task-specific predictions
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(enhanced_features)
        
        return predictions


class AffectiveClassifierHead(nn.Module):
    """Specialized classifier head for affective classification"""
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 4
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Affective-specific processing layers
        self.affective_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Affective attention mechanism
        self.affective_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Affective-specific projections
        self.affective_projections = nn.ModuleDict({
            'sarcasm': nn.Linear(hidden_dim, hidden_dim // 4),
            'persuasion': nn.Linear(hidden_dim, hidden_dim // 4),
            'harm': nn.Linear(hidden_dim, hidden_dim // 4),
            'neutral': nn.Linear(hidden_dim, hidden_dim // 4)
        })
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass for affective classification"""
        
        # Process affective features
        affective_features = self.affective_processor(embeddings)
        
        # Apply affective attention
        attended_features, _ = self.affective_attention(
            affective_features.unsqueeze(1),
            affective_features.unsqueeze(1),
            affective_features.unsqueeze(1)
        )
        
        # Combine features
        combined_features = self.layer_norm(affective_features + attended_features.squeeze(1))
        
        # Generate predictions
        predictions = self.classifier(combined_features)
        
        return predictions


class CAPMemeClassifier(nn.Module):
    """Main classifier for CAPMeme model"""
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 4,
        use_multi_task: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_multi_task = use_multi_task
        
        if self.use_multi_task:
            self.classifier = MultiTaskClassifierHead(
                input_dim, hidden_dim, num_classes
            )
        else:
            self.classifier = AffectiveClassifierHead(
                input_dim, hidden_dim, num_classes
            )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for classification"""
        
        if self.use_multi_task:
            predictions = self.classifier(embeddings)
        else:
            predictions = {'affective': self.classifier(embeddings)}
        
        # Estimate confidence
        confidence = self.confidence_head(embeddings)
        
        # Add confidence to predictions
        predictions['confidence'] = confidence
        
        return predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute classification loss"""
        
        if loss_weights is None:
            loss_weights = {
                'sarcasm': 0.25,
                'persuasion': 0.25,
                'harm': 0.25,
                'affective': 0.25
            }
        
        losses = {}
        total_loss = 0.0
        
        for task_name, pred in predictions.items():
            if task_name == 'confidence':
                continue
                
            if task_name in labels:
                if task_name == 'affective':
                    # Multi-class cross-entropy
                    loss = F.cross_entropy(pred, labels[task_name])
                else:
                    # Binary cross-entropy
                    loss = F.cross_entropy(pred, labels[task_name])
                
                losses[f'{task_name}_loss'] = loss
                total_loss += loss_weights.get(task_name, 0.25) * loss
        
        losses['total_classification_loss'] = total_loss
        
        return losses


class EnsembleClassifier(nn.Module):
    """Ensemble classifier combining multiple classification strategies"""
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 4,
        num_models: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_models = num_models
        
        # Multiple classifier models
        self.classifiers = nn.ModuleList([
            CAPMemeClassifier(input_dim, hidden_dim, num_classes, use_multi_task=True)
            for _ in range(num_models)
        ])
        
        # Ensemble fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Attention weights for ensemble
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, num_models),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for ensemble classification"""
        
        # Get predictions from all models
        all_predictions = []
        for classifier in self.classifiers:
            pred = classifier(embeddings)
            all_predictions.append(pred['affective'])
        
        # Compute attention weights
        attention_weights = self.attention_weights(embeddings)
        
        # Weighted ensemble
        weighted_predictions = []
        for i, pred in enumerate(all_predictions):
            weighted_pred = pred * attention_weights[:, i:i+1]
            weighted_predictions.append(weighted_pred)
        
        # Combine predictions
        ensemble_pred = torch.stack(weighted_predictions, dim=1)
        ensemble_pred = ensemble_pred.view(ensemble_pred.size(0), -1)
        
        # Final fusion
        final_prediction = self.fusion_layer(ensemble_pred)
        
        return {
            'affective': final_prediction,
            'individual_predictions': all_predictions,
            'attention_weights': attention_weights
        }
