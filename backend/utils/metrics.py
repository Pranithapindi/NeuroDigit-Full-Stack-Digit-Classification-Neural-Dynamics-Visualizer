import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def compute_classification_metrics(y_true, y_pred, num_classes=10):
    """
    Computes comprehensive multiclass classification metrics:
    - Overall accuracy
    - Macro and weighted Precision, Recall, F1
    - Per-class metrics (Accuracy, Precision, Recall, F1, Support)
    - 10x10 Confusion Matrix formatted for charts
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    acc = float(accuracy_score(y_true, y_pred))
    
    # Per-class precision, recall, f1, support
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    # 10x10 Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Macro averages
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    
    per_class = []
    for c in range(num_classes):
        # Class accuracy: True Positives / Total true instances for this class
        total_class = int(supp[c])
        tp = int(cm[c, c])
        class_acc = float(tp / total_class) if total_class > 0 else 0.0
        
        per_class.append({
            "class": int(c),
            "label": f"Digit {c}",
            "accuracy": round(class_acc, 4),
            "precision": round(float(prec[c]), 4),
            "recall": round(float(rec[c]), 4),
            "f1": round(float(f1[c]), 4),
            "support": total_class,
            "correct": tp
        })

    # Format confusion matrix for UI display (matrix format and heatmap records)
    cm_records = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            cm_records.append({
                "true": true_idx,
                "predicted": pred_idx,
                "count": int(cm[true_idx, pred_idx])
            })

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_records": cm_records,
        "per_class": per_class,
        "total_evaluated": len(y_true),
        "total_correct": int(np.sum(y_true == y_pred))
    }
