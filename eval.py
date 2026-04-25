import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_dice_score(pred, target, eps=1e-8):
    """
    Compute Dice coefficient for a binary foreground mask.
    """
    p = pred.astype(bool)
    t = target.astype(bool)
    
    intersection = np.logical_and(p, t).sum()
    sum_total = p.sum() + t.sum()
    
    # If the structure isn't present and the model correctly predicts none: 1.0
    if sum_total == 0:
        return 1.0
        
    return (2. * intersection) / (sum_total + eps)

def generate_metrics(y_true, y_pred, ood_true=None, ood_pred=None):
    """
    Generate binary segmentation metrics.
    """
    dices = [calculate_dice_score(p, t) for p, t in zip(y_pred, y_true)]
    
    metrics = {
        'dice_mean': float(np.mean(dices)),
        'dice_std': float(np.std(dices)),
    }
    
    if ood_true is not None and ood_pred is not None:
        metrics['ood_auc'] = float(roc_auc_score(ood_true, ood_pred))
        metrics['ood_ap'] = float(average_precision_score(ood_true, ood_pred))
        
    return metrics
