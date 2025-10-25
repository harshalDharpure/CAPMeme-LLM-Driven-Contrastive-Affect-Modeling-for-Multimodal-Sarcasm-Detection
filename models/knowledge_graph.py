"""
Knowledge Graph Integration for CAPMeme
Integrates ConceptNet and COMET for cultural and contextual understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import json
import requests
from collections import defaultdict
import pickle
import os


class ConceptNetProcessor:
    """Processor for ConceptNet knowledge graph"""
    
    def __init__(self, cache_dir: str = "data/conceptnet_cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "conceptnet_cache.pkl")
        self.cache = self._load_cache()
        
        # ConceptNet API endpoint
        self.api_url = "http://api.conceptnet.io/query"
        
        # Relevant relations for affective understanding
        self.affective_relations = {
            'Causes', 'CausesDesire', 'HasProperty', 'UsedFor',
            'CapableOf', 'Desires', 'MotivatedByGoal', 'HasA',
            'PartOf', 'LocatedNear', 'SimilarTo', 'RelatedTo'
        }
    
    def _load_cache(self) -> Dict:
        """Load cached ConceptNet data"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_cache(self):
        """Save ConceptNet data to cache"""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def query_conceptnet(self, text: str, max_relations: int = 10) -> List[Dict]:
        """Query ConceptNet for relations related to the input text"""
        # Check cache first
        cache_key = f"{text}_{max_relations}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Query ConceptNet API
            params = {
                'start': f'/c/en/{text.lower().replace(" ", "_")}',
                'limit': max_relations,
                'other': '/c/en'
            }
            
            response = requests.get(self.api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                relations = []
                
                for edge in data.get('edges', []):
                    relation = {
                        'relation': edge.get('rel', ''),
                        'target': edge.get('end', {}).get('label', ''),
                        'weight': edge.get('weight', 0.0)
                    }
                    if relation['relation'] in self.affective_relations:
                        relations.append(relation)
                
                # Cache the results
                self.cache[cache_key] = relations
                self._save_cache()
                return relations
        except Exception as e:
            print(f"Error querying ConceptNet: {e}")
        
        return []
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction (can be enhanced with NLP tools)
        words = text.lower().split()
        concepts = []
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        return concepts


class COMETProcessor:
    """Processor for COMET knowledge graph"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        # Note: In practice, you would load a COMET model here
        # For this implementation, we'll simulate COMET functionality
        
        # Common COMET relations
        self.comet_relations = {
            'xIntent': 'intent',
            'xNeed': 'need', 
            'xWant': 'want',
            'xEffect': 'effect',
            'xReact': 'reaction'
        }
    
    def generate_comet_relations(self, text: str) -> Dict[str, List[str]]:
        """Generate COMET-style relations for the input text"""
        # Simulate COMET generation (in practice, use actual COMET model)
        relations = {}
        
        # Simple heuristic-based relation generation
        if 'buy' in text.lower() or 'purchase' in text.lower():
            relations['xIntent'] = ['to purchase something', 'to buy a product']
            relations['xNeed'] = ['money', 'payment method']
            relations['xWant'] = ['to own the product', 'to get value']
        
        if 'sarcastic' in text.lower() or 'ironic' in text.lower():
            relations['xIntent'] = ['to mock something', 'to be ironic']
            relations['xEffect'] = ['confusion', 'amusement']
            relations['xReact'] = ['laugh', 'understand the irony']
        
        if 'harmful' in text.lower() or 'hurt' in text.lower():
            relations['xIntent'] = ['to cause harm', 'to hurt someone']
            relations['xEffect'] = ['emotional pain', 'distress']
            relations['xReact'] = ['feel hurt', 'be upset']
        
        return relations


class KnowledgeGraphEmbedder(nn.Module):
    """Neural network for embedding knowledge graph information"""
    
    def __init__(self, embedding_dim: int = 512, kg_embedding_dim: int = 200):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kg_embedding_dim = kg_embedding_dim
        
        # ConceptNet embedding layers
        self.conceptnet_encoder = nn.Sequential(
            nn.Linear(kg_embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
        
        # COMET embedding layers
        self.comet_encoder = nn.Sequential(
            nn.Linear(kg_embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Affective-specific knowledge processing
        self.affective_knowledge = nn.ModuleDict({
            'sarcasm': nn.Linear(embedding_dim, embedding_dim),
            'persuasion': nn.Linear(embedding_dim, embedding_dim),
            'harm': nn.Linear(embedding_dim, embedding_dim),
            'neutral': nn.Linear(embedding_dim, embedding_dim)
        })
    
    def forward(
        self,
        conceptnet_features: torch.Tensor,
        comet_features: torch.Tensor,
        affective_labels: torch.Tensor
    ) -> torch.Tensor:
        """Process knowledge graph features"""
        
        # Encode ConceptNet features
        conceptnet_encoded = self.conceptnet_encoder(conceptnet_features)
        
        # Encode COMET features
        comet_encoded = self.comet_encoder(comet_features)
        
        # Fuse knowledge features
        fused_knowledge = torch.cat([conceptnet_encoded, comet_encoded], dim=-1)
        knowledge_embeddings = self.fusion_layer(fused_knowledge)
        
        # Apply affective-specific processing
        affective_outputs = []
        for i, label in enumerate(affective_labels):
            affective_type = ['sarcasm', 'persuasion', 'harm', 'neutral'][label.item()]
            affective_output = self.affective_knowledge[affective_type](knowledge_embeddings[i:i+1])
            affective_outputs.append(affective_output)
        
        affective_knowledge = torch.cat(affective_outputs, dim=0)
        
        return affective_knowledge


class KnowledgeGraphIntegration(nn.Module):
    """Main knowledge graph integration module"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        kg_embedding_dim: int = 200,
        max_relations: int = 10,
        use_conceptnet: bool = True,
        use_comet: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kg_embedding_dim = kg_embedding_dim
        self.max_relations = max_relations
        self.use_conceptnet = use_conceptnet
        self.use_comet = use_comet
        
        # Initialize processors
        if self.use_conceptnet:
            self.conceptnet_processor = ConceptNetProcessor()
        
        if self.use_comet:
            self.comet_processor = COMETProcessor()
        
        # Knowledge embedder
        self.knowledge_embedder = KnowledgeGraphEmbedder(embedding_dim, kg_embedding_dim)
        
        # Text-to-knowledge mapping
        self.text_to_kg_projection = nn.Linear(embedding_dim, kg_embedding_dim)
        
        # Knowledge-to-text mapping
        self.kg_to_text_projection = nn.Linear(kg_embedding_dim, embedding_dim)
    
    def extract_knowledge_features(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract knowledge features from texts"""
        batch_size = len(texts)
        
        # Initialize feature tensors
        conceptnet_features = torch.zeros(batch_size, self.kg_embedding_dim)
        comet_features = torch.zeros(batch_size, self.kg_embedding_dim)
        
        for i, text in enumerate(texts):
            # Extract ConceptNet features
            if self.use_conceptnet:
                concepts = self.conceptnet_processor.extract_concepts(text)
                conceptnet_relations = []
                
                for concept in concepts[:3]:  # Limit to top 3 concepts
                    relations = self.conceptnet_processor.query_conceptnet(concept, self.max_relations)
                    conceptnet_relations.extend(relations)
                
                # Convert relations to features (simplified)
                if conceptnet_relations:
                    # Use relation weights as features
                    weights = [rel['weight'] for rel in conceptnet_relations[:self.kg_embedding_dim]]
                    if len(weights) < self.kg_embedding_dim:
                        weights.extend([0.0] * (self.kg_embedding_dim - len(weights)))
                    conceptnet_features[i] = torch.tensor(weights[:self.kg_embedding_dim])
            
            # Extract COMET features
            if self.use_comet:
                comet_relations = self.comet_processor.generate_comet_relations(text)
                
                # Convert COMET relations to features (simplified)
                comet_feature_vector = torch.zeros(self.kg_embedding_dim)
                feature_idx = 0
                
                for relation_type, relations in comet_relations.items():
                    for relation in relations:
                        if feature_idx < self.kg_embedding_dim:
                            # Simple feature encoding based on relation content
                            comet_feature_vector[feature_idx] = len(relation.split()) / 10.0
                            feature_idx += 1
                
                comet_features[i] = comet_feature_vector
        
        return conceptnet_features, comet_features
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        texts: List[str],
        affective_labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for knowledge graph integration"""
        
        # Extract knowledge features
        conceptnet_features, comet_features = self.extract_knowledge_features(texts)
        
        # Move to same device as text_embeddings
        device = text_embeddings.device
        conceptnet_features = conceptnet_features.to(device)
        comet_features = comet_features.to(device)
        
        # Process knowledge features
        knowledge_embeddings = self.knowledge_embedder(
            conceptnet_features, comet_features, affective_labels
        )
        
        # Align knowledge with text embeddings
        aligned_knowledge = self.kg_to_text_projection(
            self.text_to_kg_projection(text_embeddings)
        )
        
        # Combine text and knowledge embeddings
        enhanced_embeddings = text_embeddings + knowledge_embeddings + aligned_knowledge
        
        return enhanced_embeddings


class CulturalContextProcessor(nn.Module):
    """Process cultural and contextual cues from knowledge graphs"""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Cultural context layers
        self.cultural_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Contextual attention
        self.context_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Process cultural and contextual information"""
        
        # Encode cultural context
        cultural_features = self.cultural_encoder(embeddings)
        
        # Apply contextual attention
        attended_features, _ = self.context_attention(
            cultural_features.unsqueeze(1),
            cultural_features.unsqueeze(1),
            cultural_features.unsqueeze(1)
        )
        
        # Combine original and attended features
        enhanced_features = embeddings + attended_features.squeeze(1)
        
        return enhanced_features
