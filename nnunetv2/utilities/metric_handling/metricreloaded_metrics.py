from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures as MCPM
import numpy as np 

def macro_f1_score(pred, ref, class_list):
    """
    Calculates the macro F1-score by averaging F1-scores across all classes.
    The F1-score for each class is calculated using the binary F1-score formula,
    treating each class as a binary classification problem.
    
    Returns:
        float: Macro F1-score, the arithmetic mean of F1-scores for all classes
    """
    f1_scores = []
    for class_value in class_list:
        # Convert to binary for this class

        # pred = pred.detach().cpu().numpy()
        # ref = ref.detach().cpu().numpy()

        pred_binary = np.where(pred == class_value, 1, 0)
        ref_binary = np.where(ref == class_value, 1, 0)
        
        # Calculate F1-score for this class using BinaryPairwiseMeasures
        bpm = BPM(pred_binary, ref_binary, dict_args={"beta": 1})
        f1_score = bpm.fbeta()
        
        # Only include if the class has any reference samples
        if np.sum(ref_binary) > 0:
            f1_scores.append(f1_score)
    
    # Return average of F1-scores, or 0 if no valid F1-scores were calculated
    return np.mean(f1_scores) if f1_scores else 0

def dsc(pred, ref):
    """
    Calculates the Dice score for a given prediction and reference.
    """
    bpm = BPM(pred, ref)
    return bpm.dsc()

def micro_dice_score(pred, ref, class_list):
    """
    Calculates the micro Dice score by considering all classes together.
    This is equivalent to calculating the overall DSC across all classes.
    
    Returns:
        float: Micro Dice score
    """

    # Calculate total true positives, false positives, and false negatives
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for class_value in class_list:
        # Convert to binary for this class
        pred_binary = np.where(pred == class_value, 1, 0)
        ref_binary = np.where(ref == class_value, 1, 0)
        
        # Calculate TP, FP, FN for this class
        tp = np.sum(pred_binary * ref_binary)
        fp = np.sum(pred_binary * (1 - ref_binary))
        fn = np.sum((1 - pred_binary) * ref_binary)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate micro DSC
    numerator = 2 * total_tp
    denominator = 2 * total_tp + total_fp + total_fn
    
    if denominator == 0:
        return 0.0
    return numerator / denominator