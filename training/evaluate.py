import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import LABELS


def labels_to_int(y):
    y = np.asarray(y)
    if y.ndim > 1:
        return np.argmax(y, axis=1)
    return y.astype(np.int32).ravel()


def evaluate_model(model, X, y, batch_size=512, print_report=True):
    y_prob = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = labels_to_int(y)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )

    if print_report:
        print(f"Accuracy: {accuracy:.10f}")
        print(f"Macro F1: {macro_f1:.10f}")
        print("\nClassification report:")
        print(report)
        print("Confusion matrix:")
        print(cm)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm,
        "classification_report": report,
    }
