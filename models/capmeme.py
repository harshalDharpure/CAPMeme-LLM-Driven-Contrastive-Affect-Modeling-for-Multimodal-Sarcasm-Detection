"""
CAPMeme: Main Model Implementation
Integrates all components for multimodal sarcasm and persuasive meme detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import yaml
import os

# Import all model components
from models.vision_language_encoder import VisionLanguageEncoder
from models.text_encoder import TextEncoder, AffectiveTextProcessor
from models.contrastive_pretraining import ContrastiveAffectivePretraining, MultiModalFusion
from models.knowledge_graph import KnowledgeGraphIntegration, CulturalContextProcessor
from models.classifier_head import CAPMemeClassifier


class CAPMeme(nn.Module):
    """
    CAPMeme: Multimodal Sarcasm and Persuasive Meme Detection Model
    
    This model integrates:
    - Vision-language encoding (CLIP/BLIP)
    - Deep textual understanding (BERT/LLaMA)
    - Contrastive Affective Pretraining (CAP)
    - Knowledge graph integration (ConceptNet/COMET)
    - Multi-task classification
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model configuration
        self.embedding_dim = config['model']['embedding_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.num_classes = config['model']['num_classes']
        
        # Initialize components
        self._initialize_components()
        
        # Training mode flags
        self.training_mode = 'full'  # 'pretraining', 'classification', 'full'
    
    def _initialize_components(self):
        """Initialize all model components"""
        
        # Vision-language encoder
        self.vision_language_encoder = VisionLanguageEncoder(
            encoder_type=self.config['model']['vision_encoder'],
            embedding_dim=self.embedding_dim
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            encoder_type=self.config['model']['text_encoder'],
            embedding_dim=self.embedding_dim
        )
        
        # Affective text processor
        self.affective_text_processor = AffectiveTextProcessor(
            embedding_dim=self.embedding_dim
        )
        
        # Contrastive Affective Pretraining
        self.cap_module = ContrastiveAffectivePretraining(
            embedding_dim=self.embedding_dim,
            temperature=self.config['cap']['temperature'],
            contrastive_weight=self.config['cap']['contrastive_weight'],
            affective_weight=self.config['cap']['affective_weight'],
            knowledge_weight=self.config['cap']['knowledge_weight']
        )
        
        # Knowledge graph integration
        self.knowledge_graph = KnowledgeGraphIntegration(
            embedding_dim=self.embedding_dim,
            kg_embedding_dim=self.config['knowledge_graph']['kg_embedding_dim'],
            max_relations=self.config['knowledge_graph']['max_relations'],
            use_conceptnet=self.config['knowledge_graph']['use_conceptnet'],
            use_comet=self.config['knowledge_graph']['use_comet']
        )
        
        # Cultural context processor
        self.cultural_context_processor = CulturalContextProcessor(
            embedding_dim=self.embedding_dim
        )
        
        # Multi-modal fusion
        self.multimodal_fusion = MultiModalFusion(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Classifier head
        self.classifier = CAPMemeClassifier(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            use_multi_task=True
        )
        
        # Feature projection layers
        self.feature_projector = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str],
        affective_labels: Optional[torch.Tensor] = None,
        sarcasm_labels: Optional[torch.Tensor] = None,
        persuasion_labels: Optional[torch.Tensor] = None,
        harm_labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Forward pass through the CAPMeme model
        
        Args:
            images: Input images [batch_size, channels, height, width]
            input_ids: Text input IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            texts: List of text strings
            affective_labels: Affective labels for CAP training
            sarcasm_labels: Sarcasm labels
            persuasion_labels: Persuasion labels
            harm_labels: Harm labels
            return_embeddings: Whether to return intermediate embeddings
        
        Returns:
            Model outputs and optionally embeddings
        """
        
        # Extract vision-language embeddings
        image_embeddings, vision_text_embeddings = self.vision_language_encoder(
            images, input_ids, attention_mask
        )
        
        # Extract deep textual embeddings
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        
        # Process affective text features
        affective_features = self.affective_text_processor(text_embeddings)
        
        # Integrate knowledge graph information
        if self.training_mode in ['pretraining', 'full']:
            enhanced_text_embeddings = self.knowledge_graph(
                text_embeddings, texts, affective_labels
            )
        else:
            enhanced_text_embeddings = text_embeddings
        
        # Process cultural context
        enhanced_text_embeddings = self.cultural_context_processor(enhanced_text_embeddings)
        
        # Fuse multimodal embeddings
        if affective_labels is not None:
            fused_embeddings = self.multimodal_fusion(
                image_embeddings, enhanced_text_embeddings, 
                torch.zeros_like(image_embeddings), affective_labels
            )
        else:
            # Simple concatenation if no affective labels
            fused_embeddings = torch.cat([image_embeddings, enhanced_text_embeddings], dim=-1)
            fused_embeddings = self.feature_projector(fused_embeddings)
        
        # Generate predictions
        predictions = self.classifier(fused_embeddings)
        
        # Prepare outputs
        outputs = {
            'predictions': predictions,
            'fused_embeddings': fused_embeddings
        }
        
        # Compute CAP losses if in pretraining mode
        if self.training_mode in ['pretraining', 'full'] and affective_labels is not None:
            cap_losses = self.cap_module(
                image_embeddings, enhanced_text_embeddings,
                affective_labels, sarcasm_labels, persuasion_labels, harm_labels
            )
            outputs['cap_losses'] = cap_losses
        
        # Return embeddings if requested
        if return_embeddings:
            embeddings = {
                'image_embeddings': image_embeddings,
                'text_embeddings': text_embeddings,
                'enhanced_text_embeddings': enhanced_text_embeddings,
                'affective_features': affective_features,
                'fused_embeddings': fused_embeddings
            }
            return outputs, embeddings
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss for training"""
        
        if loss_weights is None:
            loss_weights = {
                'classification': 0.6,
                'contrastive': 0.2,
                'affective': 0.1,
                'knowledge': 0.1
            }
        
        total_loss = 0.0
        losses = {}
        
        # Classification loss
        if 'predictions' in outputs:
            classification_losses = self.classifier.compute_loss(
                outputs['predictions'], labels
            )
            losses.update(classification_losses)
            total_loss += loss_weights['classification'] * classification_losses['total_classification_loss']
        
        # CAP losses
        if 'cap_losses' in outputs:
            cap_losses = outputs['cap_losses']
            losses.update(cap_losses)
            
            total_loss += loss_weights['contrastive'] * cap_losses['contrastive_loss']
            total_loss += loss_weights['affective'] * cap_losses['affective_loss']
            total_loss += loss_weights['knowledge'] * cap_losses['knowledge_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def set_training_mode(self, mode: str):
        """Set training mode: 'pretraining', 'classification', or 'full'"""
        self.training_mode = mode
        
        if mode == 'pretraining':
            # Freeze classifier, train CAP components
            for param in self.classifier.parameters():
                param.requires_grad = False
            for param in self.cap_module.parameters():
                param.requires_grad = True
        elif mode == 'classification':
            # Freeze CAP components, train classifier
            for param in self.cap_module.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True
        else:  # 'full'
            # Train all components
            for param in self.parameters():
                param.requires_grad = True
    
    def get_embeddings(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Extract embeddings for analysis"""
        self.eval()
        with torch.no_grad():
            _, embeddings = self.forward(
                images, input_ids, attention_mask, texts,
                return_embeddings=True
            )
        return embeddings
    
    def predict(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Make predictions on new data"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                images, input_ids, attention_mask, texts
            )
        return outputs['predictions']
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_mode': self.training_mode
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_mode = checkpoint.get('training_mode', 'full')
    
    @classmethod
    def from_config(cls, config_path: str):
        """Create model from configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)


class CAPMemeTrainer:
    """Training wrapper for CAPMeme model - Optimized for GPU training"""
    
    def __init__(self, model: CAPMeme, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Enable mixed precision training for GPU efficiency
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using Automatic Mixed Precision (AMP) for GPU training")
        
        # Initialize optimizer with better settings for large dataset
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['training']['learning_rate'],
            epochs=config['training']['epochs'],
            steps_per_epoch=1,  # Will be updated in training loop
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def train_epoch(self, train_loader, evaluator):
        """Train for one epoch with GPU optimizations"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Update scheduler steps_per_epoch
        self.scheduler.step()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device efficiently
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        images=batch['image'],
                        input_ids=batch['text'],
                        attention_mask=batch['attention_mask'],
                        texts=[f"text_{i}" for i in range(len(batch['image']))],
                        affective_labels=batch['affective_label'],
                        sarcasm_labels=batch['sarcasm_label'],
                        persuasion_labels=batch['persuasion_label'],
                        harm_labels=batch['harm_label']
                    )
                    
                    # Compute loss
                    labels = {
                        'affective': batch['affective_label'],
                        'sarcasm': batch['sarcasm_label'],
                        'persuasion': batch['persuasion_label'],
                        'harm': batch['harm_label']
                    }
                    
                    losses = self.model.compute_loss(outputs, labels)
            else:
                outputs = self.model(
                    images=batch['image'],
                    input_ids=batch['text'],
                    attention_mask=batch['attention_mask'],
                    texts=[f"text_{i}" for i in range(len(batch['image']))],
                    affective_labels=batch['affective_label'],
                    sarcasm_labels=batch['sarcasm_label'],
                    persuasion_labels=batch['persuasion_label'],
                    harm_labels=batch['harm_label']
                )
                
                # Compute loss
                labels = {
                    'affective': batch['affective_label'],
                    'sarcasm': batch['sarcasm_label'],
                    'persuasion': batch['persuasion_label'],
                    'harm': batch['harm_label']
                }
                
                losses = self.model.compute_loss(outputs, labels)
            
            # Backward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(losses['total_loss']).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
                
                self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            num_batches += 1
            
            # Clear cache periodically to prevent memory issues
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader, evaluator):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    images=batch['image'],
                    input_ids=batch['text'],
                    attention_mask=batch['attention_mask'],
                    texts=[f"text_{i}" for i in range(len(batch['image']))],
                    affective_labels=batch['affective_label'],
                    sarcasm_labels=batch['sarcasm_label'],
                    persuasion_labels=batch['persuasion_label'],
                    harm_labels=batch['harm_label']
                )
                
                # Compute loss
                labels = {
                    'affective': batch['affective_label'],
                    'sarcasm': batch['sarcasm_label'],
                    'persuasion': batch['persuasion_label'],
                    'harm': batch['harm_label']
                }
                
                losses = self.model.compute_loss(outputs, labels)
                total_loss += losses['total_loss'].item()
                
                # Collect predictions for evaluation
                predictions = outputs['predictions']['affective']
                all_predictions.append(predictions)
                all_labels.append(batch['affective_label'])
        
        # Concatenate predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = evaluator.evaluate_batch(all_predictions, all_labels)
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics
