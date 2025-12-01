# src/metrics.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import math
from typing import Dict

def classification_metrics(preds, labels) -> Dict:
    """Compute accuracy, precision, recall, and f1 (weighted). 
    preds can be logits (ndarray) or 1d predicted labels."""
    import numpy as _np
    preds_arr = _np.array(preds)
    if preds_arr.ndim > 1:
        pred_labels = preds_arr.argmax(axis=1)
    else:
        pred_labels = preds_arr
    acc = accuracy_score(labels, pred_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred_labels, average='weighted', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def perplexity(loss):
    return math.exp(loss)
