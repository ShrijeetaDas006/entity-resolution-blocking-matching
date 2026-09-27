def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)

def evaluate_pipeline_predictions(pred_pairs: set, gt_pairs: set, total_possible_pairs: int):
    tp = len(pred_pairs.intersection(gt_pairs))
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f0_5 = calculate_f_beta(precision, recall, beta=0.5)
    
    reduction_ratio = 1.0 - (len(pred_pairs) / total_possible_pairs) if total_possible_pairs > 0 else 0.0

    return {
        "Candidate Pairs": len(pred_pairs),
        "True Positives": tp,
        "Precision": f"{precision:.4%}",
        "Recall": f"{recall:.4%}",
        "F0.5 Score": f"{f0_5:.4%}",
        "Reduction Ratio": f"{reduction_ratio:.4%}"
    }