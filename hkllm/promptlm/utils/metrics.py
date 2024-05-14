from collections import Counter
def safe_divide(num, den):
    """Prevent division by zero errors by returning 0 if the denominator is 0."""
    return num / den if den else 0





def calculate_metrics(y_true, y_pred, average='macro', verbosity=0, include_specificity=False, include_npv=False):
    if len(y_true) != len(y_pred):
        raise ValueError("Predictions and ground truth must be lists of the same length")

    classes = set(y_true) | set(y_pred)
    metrics = {}
    total_instances = len(y_true)
    class_weights = Counter(y_true)

    # Initialize counts for true positives, false positives, false negatives, and true negatives
    sum_tp, sum_fp, sum_fn, sum_tn = 0, 0, 0, 0

    # Calculate per-class metrics
    for cls in classes:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == cls)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        tn = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred != cls)

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        sum_tn += tn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        class_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': class_weights[cls],
            'specificity': specificity if include_specificity else None,
            'npv': npv if include_npv else None
        }
        metrics[cls] = class_metrics

    # Calculate overall metrics based on averaging method
    if average == 'micro':
        overall_accuracy = (sum_tp + sum_tn) / total_instances
        overall_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
        overall_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_metrics = {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        }
    elif average == 'macro':
        overall_metrics = {
            'accuracy': sum(m['accuracy'] for m in metrics.values()) / len(classes),
            'precision': sum(m['precision'] for m in metrics.values()) / len(classes),
            'recall': sum(m['recall'] for m in metrics.values()) / len(classes),
            'f1': sum(m['f1'] for m in metrics.values()) / len(classes)
        }
    elif average == 'weighted':
        overall_metrics = {
            'accuracy': sum(m['accuracy'] * m['support'] for m in metrics.values()) / total_instances,
            'precision': sum(m['precision'] * m['support'] for m in metrics.values()) / total_instances,
            'recall': sum(m['recall'] * m['support'] for m in metrics.values()) / total_instances,
            'f1': sum(m['f1'] * m['support'] for m in metrics.values()) / total_instances
        }

    metrics['overall'] = overall_metrics

    if verbosity > 0:
        print("Detailed Metrics Computation:")
        for cls, class_metrics in metrics.items():
            print(f"Class {cls} Metrics: {class_metrics}")

    return metrics


    
    
def count_distribution(y_true, y_pred):
    actual_label_distributions = Counter(y_true)
    pred_label_distribution = Counter(y_pred)
    
    return actual_label_distributions,pred_label_distribution




  
def calculate_confusion_matrix_elements(y_true, y_pred, positive_classes):
    # Initialize counts
    TP = FP = TN = FN = 0

    for true, pred in zip(y_true, y_pred):
        if true in positive_classes and pred in positive_classes:
            TP += 1
        elif true not in positive_classes and pred in positive_classes:
            FP += 1
        elif true not in positive_classes and pred not in positive_classes:
            TN += 1
        elif true in positive_classes and pred not in positive_classes:
            FN += 1

    return TP, FP, TN, FN

def calculate_binary_metrics(y_true, y_pred, positive_classes):
    TP, FP, TN, FN = calculate_confusion_matrix_elements(y_true, y_pred, positive_classes)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    
    return accuracy, precision, recall, f1


