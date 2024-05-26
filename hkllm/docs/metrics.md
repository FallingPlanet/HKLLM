###Documentation for 'calculate_metrics'
## `calculate_metrics`

Calculates various classification metrics such as precision, recall, F1 score, and accuracy for given true and predicted labels.

### Parameters
- **y_true** (`list` of `str`): Actual labels as strings.
- **y_pred** (`list` of `str`): Predicted labels as strings.
- **average** (`str`, optional): Type of averaging to perform ('micro', 'macro', 'weighted'). Defaults to 'macro'.
- **verbosity** (`int`, optional): Level of verbosity (0 or 1). Higher numbers increase verbosity. Defaults to 0.
- **include_specificity** (`bool`, optional): Whether to include specificity in the output. Defaults to False.
- **include_npv** (`bool`, optional): Whether to include NPV (Negative Predictive Value) in the output. Defaults to False.

### Returns
- `dict`: A dictionary containing metrics for each class and overall metrics.

### Example
```python
y_true = ["cat", "dog", "cat", "bird"]
y_pred = ["cat", "dog", "cat", "cat"]
metrics = calculate_metrics(y_true, y_pred, average='macro', verbosity=1)
# Output: Detailed metrics printed and returned as a dictionary
```

### Documentation for `count_distribution`

## `count_distribution`

Calculates the distribution of actual and predicted labels.

### Parameters
- **y_true** (`list` of `str`): Actual labels as strings.
- **y_pred** (`list` of `str`): Predicted labels as strings.

### Returns
- `tuple`: A tuple of two `collections.Counter` objects representing the distribution of actual labels and predicted labels respectively.

### Example
```python
y_true = ["cat", "dog", "cat", "bird"]
y_pred = ["cat", "dog", "cat", "cat"]
actual_dist, pred_dist = count_distribution(y_true, y_pred)
# Output: (Counter({'cat': 2, 'dog': 1, 'bird': 1}), Counter({'cat': 3, 'dog': 1}))
```

### Documentation for `calculate_confusion_matrix_elements`

## `calculate_confusion_matrix_elements`

Calculates the elements of a confusion matrix (TP, FP, TN, FN) for given true and predicted labels based on specified positive classes.

### Parameters
- **y_true** (`list` of `str`): Actual labels as strings.
- **y_pred** (`list` of `str`): Predicted labels as strings.
- **positive_classes** (`list` of `str`): Classes considered as positive.

### Returns
- `tuple`: A tuple containing the counts of TP, FP, TN, and FN.

### Example
```python
y_true = ["cat", "dog", "cat", "bird"]
y_pred = ["cat", "dog", "cat", "cat"]
positive_classes = ["cat"]
TP, FP, TN, FN = calculate_confusion_matrix_elements(y_true, y_pred, positive_classes)
# Output: (2, 1, 1, 1)
```

### Documentation for `calculate_binary_metrics`


## `calculate_binary_metrics`

Calculates binary classification metrics such as accuracy, precision, recall, and F1 score based on the confusion matrix elements for specified positive classes.

### Parameters
- **y_true** (`list` of `str`): Actual labels as strings.
- **y_pred** (`list` of `str`): Predicted labels as strings.
- **positive_classes** (`list` of `str`): Classes considered as positive.

### Returns
- `tuple`: A tuple containing the calculated accuracy, precision, recall, and F1 score.

### Example
```python
y_true = ["cat", "dog", "cat", "bird"]
y_pred = ["cat", "dog", "cat", "cat"]
positive_classes = ["cat"]
accuracy, precision, recall, f1 = calculate_binary_metrics(y_true, y_pred, positive_classes)
# Output: (0.75, 0.666, 1.0, 0.8)
```
