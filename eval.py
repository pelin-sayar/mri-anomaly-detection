import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_dice_score(pred, target, class_idx, eps=1e-8):
    """
    Compute Dice coefficient for a specific class in a multi-class mask.
    0=Background, 1=Aorta, 2=Artery
    """
    p = (pred == class_idx)
    t = (target == class_idx)
    
    intersection = np.logical_and(p, t).sum()
    sum_total = p.sum() + t.sum()
    
    # If the artery isn't in the slice and the model correctly predicts none: 1.0
    if sum_total == 0:
        return 1.0
        
    return (2. * intersection) / (sum_total + eps)

def generate_metrics(y_true, y_pred, ood_true=None, ood_pred=None):
    """
    Generate detailed clinical metrics with a focus on Artery performance.
    """
    # 1. Calculate class-specific scores
    aorta_dices = [calculate_dice_score(p, t, class_idx=1) for p, t in zip(y_pred, y_true)]
    artery_dices = [calculate_dice_score(p, t, class_idx=2) for p, t in zip(y_pred, y_true)]
    
    # 2. Compile metrics dictionary
    metrics = {
        # Clinical Focus: The Artery is your primary KPI
        'dice_artery_mean': float(np.mean(artery_dices)),
        'dice_artery_std': float(np.std(artery_dices)),
        
        # Structural Context: How well we see the Aorta
        'dice_aorta_mean': float(np.mean(aorta_dices)),
        
        # Combined Score
        'dice_overall_mean': float(np.mean(aorta_dices + artery_dices)),
    }
    
    # 3. OOD/Flagging Performance
    if ood_true is not None and ood_pred is not None:
        # AUC tells you how reliable your 'Flag' is across all thresholds
        metrics['ood_auc'] = float(roc_auc_score(ood_true, ood_pred))
        metrics['ood_ap'] = float(average_precision_score(ood_true, ood_pred))
        
    return metrics