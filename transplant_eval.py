import torch as t
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TransplantEvalResult:
    """
    Stores evaluation results for head transplantation experiments.
    """
    base_accuracy: float
    transplant_accuracy: float
    accuracy_delta: float
    per_class_accuracies_base: Dict[int, float]
    per_class_accuracies_transplant: Dict[int, float]
    per_class_accuracy_deltas: Dict[int, float]
    n_samples: int
    transplant_spec: List[Tuple[int, Optional[List[int]]]]

class TransplantEvaluator:
    """
    Evaluates models before and after head transplantation.
    """
    def __init__(self, n_classes: int, device: str = 'cuda' if t.cuda.is_available() else 'cpu'):
        self.n_classes = n_classes
        self.device = device

    def _get_model_predictions(self, model: t.nn.Module, dataloader) -> Tuple[t.Tensor, t.Tensor]:
        """
        Get model predictions for the entire dataset.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader containing the evaluation data
            
        Returns:
            Tuple of (predictions, targets)
        """
        model.eval()
        model = model.to(self.device)
        
        all_preds = []
        all_targets = []
        
        with t.no_grad():
            for input_strings, inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get model outputs
                outputs = model(inputs)[0]
                
                # Calculate sequence lengths (including special tokens)
                seq_lens = t.tensor([len(s) + 2 for s in input_strings])
                
                # Get predictions at the last relevant position for each sequence
                last_indices = (seq_lens - 1).unsqueeze(1).repeat(1, self.n_classes)
                last_indices = last_indices.to(self.device)
                preds = outputs.gather(1, last_indices.unsqueeze(1))[:, 0, :]
                
                # Get predicted classes
                _, predicted = t.max(preds, 1)
                
                all_preds.append(predicted)
                all_targets.append(targets)
        
        return t.cat(all_preds), t.cat(all_targets)

    def _compute_per_class_accuracy(self, 
                                  predictions: t.Tensor, 
                                  targets: t.Tensor) -> Dict[int, float]:
        """
        Compute accuracy for each class separately.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Dictionary mapping class indices to their accuracies
        """
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        for pred, target in zip(predictions, targets):
            per_class_total[target.item()] += 1
            if pred == target:
                per_class_correct[target.item()] += 1
        
        return {
            class_idx: (per_class_correct[class_idx] / per_class_total[class_idx])
            for class_idx in per_class_total.keys()
        }

    def evaluate_transplant(self,
                          source_model: t.nn.Module,
                          target_model: t.nn.Module,
                          dataloader,
                          transplant_spec: List[Tuple[int, Optional[List[int]]]]) -> TransplantEvalResult:
        """
        Evaluate model performance before and after head transplantation.
        
        Args:
            source_model: Source model to copy heads from
            target_model: Target model to copy heads to
            dataloader: DataLoader containing evaluation data
            transplant_spec: List of (layer_idx, head_indices) tuples specifying transplantation pattern
            
        Returns:
            TransplantEvalResult containing detailed evaluation metrics
        """
        # Get baseline predictions
        base_preds, targets = self._get_model_predictions(target_model, dataloader)
        
        # Perform transplant
        target_model.transplant_attention_heads(source_model, transplant_spec)
        
        # Get post-transplant predictions
        transplant_preds, _ = self._get_model_predictions(target_model, dataloader)
        
        # Compute overall accuracies
        base_acc = (base_preds == targets).float().mean().item()
        transplant_acc = (transplant_preds == targets).float().mean().item()
        
        # Compute per-class accuracies
        per_class_base = self._compute_per_class_accuracy(base_preds, targets)
        per_class_transplant = self._compute_per_class_accuracy(transplant_preds, targets)
        
        # Compute per-class accuracy deltas
        per_class_deltas = {
            class_idx: transplant_acc - base_acc
            for class_idx, (base_acc, transplant_acc) 
            in zip(per_class_base.keys(), 
                  zip(per_class_base.values(), per_class_transplant.values()))
        }
        
        return TransplantEvalResult(
            base_accuracy=base_acc,
            transplant_accuracy=transplant_acc,
            accuracy_delta=transplant_acc - base_acc,
            per_class_accuracies_base=per_class_base,
            per_class_accuracies_transplant=per_class_transplant,
            per_class_accuracy_deltas=per_class_deltas,
            n_samples=len(targets),
            transplant_spec=transplant_spec
        )

    def eval_comparative_transplants(self,
                                   source_models: List[t.nn.Module],
                                   target_model: t.nn.Module,
                                   dataloader,
                                   transplant_specs: List[List[Tuple[int, Optional[List[int]]]]]) -> List[TransplantEvalResult]:
        """
        Evaluate multiple transplantation patterns and/or source models.
        
        Args:
            source_models: List of source models to evaluate
            target_model: Target model to receive transplanted heads
            dataloader: DataLoader containing evaluation data
            transplant_specs: List of transplantation specifications to try
            
        Returns:
            List of TransplantEvalResult for each combination tried
        """
        results = []
        
        # Store original target model weights
        original_state = target_model.state_dict()
        
        for source_model in source_models:
            for transplant_spec in transplant_specs:
                # Restore original weights
                target_model.load_state_dict(original_state)
                
                # Evaluate this combination
                result = self.evaluate_transplant(
                    source_model, target_model, dataloader, transplant_spec
                )
                results.append(result)
        
        return results
    
## use

# evaluator = TransplantEvaluator(n_classes=N_CLASSES)
# result = evaluator.evaluate_transplant(
#     source_model=source_model,
#     target_model=target_model,
#     dataloader=test_dataloader,
#     transplant_spec=[(0, None)]  # Transplant all heads in layer 0
# )
# print(f"Accuracy change: {result.accuracy_delta:.3f}")


# results = evaluator.eval_comparative_transplants(
#     source_models=[model1, model2],
#     target_model=target_model,
#     dataloader=test_dataloader,
#     transplant_specs=[
#         [(0, None)],           # All heads in layer 0
#         [(1, [0, 1])],         # Heads 0,1 in layer 1
#         [(0, None), (1, None)] # All heads in layers 0 and 1
#     ]
# )

# # Find best transplant pattern
# best_result = max(results, key=lambda x: x.accuracy_delta)
# print(f"Best transplant improved accuracy by {best_result.accuracy_delta:.3f}")
