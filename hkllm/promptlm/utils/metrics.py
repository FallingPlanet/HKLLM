from collections import Counter
def safe_divide(num, den):
    """Prevent division by zero errors by returning 0 if the denominator is 0."""
    return num / den if den else 0

def calculate_metrics(y_true, y_pred, average='macro', verbosity=0, include_specificity=False, include_npv=False):
    """Calculate various classification metrics, allowing for different averaging methods and optional metrics."""
    if len(y_true) != len(y_pred):
        raise ValueError("Predictions and ground truth must be lists of the same length")

    classes = set(y_true) | set(y_pred)
    metrics = {}
    total_instances = len(y_true)
    class_weights = Counter(y_true)

    # Initialize sums for micro averages
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

        if average != 'micro':
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            f1 = safe_divide(2 * precision * recall, precision + recall)
            specificity = safe_divide(tn, tn + fp) if include_specificity else None
            npv = safe_divide(tn, tn + fn) if include_npv else None

            class_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': class_weights[cls]
            }
            if include_specificity:
                class_metrics['specificity'] = specificity
            if include_npv:
                class_metrics['npv'] = npv

            metrics[cls] = class_metrics

    # Calculate overall accuracy and average metrics
    accuracy = safe_divide(sum_tp + sum_tn, total_instances)

    # Micro average calculations
    if average == 'micro':
        micro_precision = safe_divide(sum_tp, sum_tp + sum_fp)
        micro_recall = safe_divide(sum_tp, sum_tp + sum_fn)
        micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        metrics['overall'] = {'accuracy': accuracy, 'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1}

    elif average == 'macro':
        macro_metrics = {metric: sum(class_metrics[metric] for class_metrics in metrics.values()) / len(classes) for metric in metrics[next(iter(metrics))]}
        macro_metrics['accuracy'] = accuracy
        metrics['overall'] = macro_metrics

    elif average == 'weighted':
        total_weight = sum(class_metrics['support'] for class_metrics in metrics.values())
        weighted_metrics = {metric: sum(class_metrics[metric] * class_metrics['support'] for class_metrics in metrics.values()) / total_weight for metric in metrics[next(iter(metrics))]}
        weighted_metrics['accuracy'] = accuracy
        metrics['overall'] = weighted_metrics

    if verbosity > 0:
        print("Detailed Metrics Computation:")
        for cls, class_metrics in metrics.items():
            if cls != 'overall':
                print(f"Class {cls} Metrics: {class_metrics}")

    return metrics['overall']
    
    
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


