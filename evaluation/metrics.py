"""
Evaluation Metrics for CAPMeme
Implements comprehensive evaluation including Accuracy, AUROC, and F1-score
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class CAPMemeEvaluator:
    """Comprehensive evaluator for CAPMeme model"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['sarcastic', 'persuasive', 'harmful', 'neutral']
        self.num_classes = len(self.class_names)
        
        # Store evaluation history
        self.evaluation_history = defaultdict(list)
    
    def compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute accuracy score"""
        pred_labels = torch.argmax(predictions, dim=-1)
        accuracy = accuracy_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
        return accuracy
    
    def compute_f1_scores(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute F1 scores (macro, micro, weighted)"""
        pred_labels = torch.argmax(predictions, dim=-1)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels.cpu().numpy(), pred_labels.cpu().numpy(), average=None
        )
        
        # Compute different F1 averages
        f1_macro = np.mean(f1)
        f1_micro = precision_recall_fscore_support(
            labels.cpu().numpy(), pred_labels.cpu().numpy(), average='micro'
        )[2]
        f1_weighted = precision_recall_fscore_support(
            labels.cpu().numpy(), pred_labels.cpu().numpy(), average='weighted'
        )[2]
        
        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1.tolist(),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist()
        }
    
    def compute_auroc(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute Area Under ROC Curve for each class"""
        predictions_np = F.softmax(predictions, dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        auroc_scores = {}
        
        # One-vs-Rest AUROC for each class
        for i, class_name in enumerate(self.class_names):
            try:
                # Create binary labels for this class
                binary_labels = (labels_np == i).astype(int)
                if len(np.unique(binary_labels)) > 1:  # Check if both classes exist
                    auroc = roc_auc_score(binary_labels, predictions_np[:, i])
                    auroc_scores[f'auroc_{class_name}'] = auroc
                else:
                    auroc_scores[f'auroc_{class_name}'] = 0.5  # Random performance
            except ValueError:
                auroc_scores[f'auroc_{class_name}'] = 0.5
        
        # Macro average AUROC
        auroc_scores['auroc_macro'] = np.mean(list(auroc_scores.values()))
        
        return auroc_scores
    
    def compute_confusion_matrix(self, predictions: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
        """Compute confusion matrix"""
        pred_labels = torch.argmax(predictions, dim=-1)
        cm = confusion_matrix(labels.cpu().numpy(), pred_labels.cpu().numpy())
        return cm
    
    def evaluate_batch(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        task_name: str = 'affective'
    ) -> Dict[str, float]:
        """Evaluate a single batch"""
        
        metrics = {}
        
        # Accuracy
        metrics[f'{task_name}_accuracy'] = self.compute_accuracy(predictions, labels)
        
        # F1 scores
        f1_scores = self.compute_f1_scores(predictions, labels)
        for key, value in f1_scores.items():
            metrics[f'{task_name}_{key}'] = value
        
        # AUROC scores
        auroc_scores = self.compute_auroc(predictions, labels)
        for key, value in auroc_scores.items():
            metrics[f'{task_name}_{key}'] = value
        
        return metrics
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        task_name: str = 'affective'
    ) -> Dict[str, float]:
        """Evaluate model on entire dataset"""
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                images = batch['image'].to(device)
                text_ids = batch['text'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch[f'{task_name}_label'].to(device)
                
                # Get model predictions
                outputs = model(images, text_ids, attention_mask)
                if isinstance(outputs, dict):
                    predictions = outputs[task_name]
                else:
                    predictions = outputs
                
                all_predictions.append(predictions)
                all_labels.append(labels)
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        metrics = self.evaluate_batch(all_predictions, all_labels, task_name)
        
        # Store evaluation history
        for key, value in metrics.items():
            self.evaluation_history[key].append(value)
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        cm = self.compute_confusion_matrix(predictions, labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves from evaluation history"""
        if not self.evaluation_history:
            print("No evaluation history available")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        if 'affective_accuracy' in self.evaluation_history:
            axes[0, 0].plot(self.evaluation_history['affective_accuracy'])
            axes[0, 0].set_title('Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
        
        # Plot F1 macro
        if 'affective_f1_macro' in self.evaluation_history:
            axes[0, 1].plot(self.evaluation_history['affective_f1_macro'])
            axes[0, 1].set_title('F1 Macro')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
        
        # Plot AUROC macro
        if 'affective_auroc_macro' in self.evaluation_history:
            axes[1, 0].plot(self.evaluation_history['affective_auroc_macro'])
            axes[1, 0].set_title('AUROC Macro')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUROC')
        
        # Plot loss (if available)
        if 'total_loss' in self.evaluation_history:
            axes[1, 1].plot(self.evaluation_history['total_loss'])
            axes[1, 1].set_title('Total Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> str:
        """Generate detailed classification report"""
        pred_labels = torch.argmax(predictions, dim=-1)
        
        report = classification_report(
            labels.cpu().numpy(),
            pred_labels.cpu().numpy(),
            target_names=self.class_names,
            digits=4
        )
        
        return report
    
    def compute_per_class_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Compute detailed per-class metrics"""
        pred_labels = torch.argmax(predictions, dim=-1)
        
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            # Create binary labels for this class
            binary_labels = (labels == i).int()
            binary_preds = (pred_labels == i).int()
            
            # Compute metrics
            tp = ((binary_preds == 1) & (binary_labels == 1)).sum().item()
            fp = ((binary_preds == 1) & (binary_labels == 0)).sum().item()
            fn = ((binary_preds == 0) & (binary_labels == 1)).sum().item()
            tn = ((binary_preds == 0) & (binary_labels == 0)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
        
        return per_class_metrics


class MultiTaskEvaluator:
    """Evaluator for multi-task learning scenarios"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['sarcastic', 'persuasive', 'harmful', 'neutral']
        self.evaluator = CAPMemeEvaluator(class_names)
    
    def evaluate_multi_task(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multi-task predictions"""
        
        results = {}
        
        for task_name, pred in predictions.items():
            if task_name in labels:
                task_results = self.evaluator.evaluate_batch(pred, labels[task_name], task_name)
                results[task_name] = task_results
        
        return results
    
    def compute_task_correlation(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute correlation between different tasks"""
        correlations = {}
        
        # Extract F1 scores for correlation analysis
        f1_scores = {}
        for task_name, metrics in results.items():
            if 'f1_macro' in metrics:
                f1_scores[task_name] = metrics['f1_macro']
        
        # Compute pairwise correlations
        task_names = list(f1_scores.keys())
        for i, task1 in enumerate(task_names):
            for j, task2 in enumerate(task_names):
                if i < j:
                    corr_key = f'{task1}_{task2}_correlation'
                    correlations[corr_key] = np.corrcoef([f1_scores[task1], f1_scores[task2]])[0, 1]
        
        return correlations
