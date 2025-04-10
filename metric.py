import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np

def generate_and_evaluate_corrected_data():
    """Generates data and evaluates it."""

    data = {
        "Known Trend": ["Down"] * 16 + ["Up"] * 56 + ["Down"] * 20,
        "Predicted Trend": ["Down"] * 13 + ["Up"] * 3  + ["Up"] * 52 + ["Down"] * 4 + ["Down"] * 18 + ["Up"] * 2 ,
    }

    df = pd.DataFrame(data)

    known_trends = df["Known Trend"].tolist()
    predicted_trends = df["Predicted Trend"].tolist()

    unique_labels = sorted(list(set(known_trends + predicted_trends)))
    label_to_num = {label: i for i, label in enumerate(unique_labels)}

    known_trends_num = [label_to_num[label] for label in known_trends]
    predicted_trends_num = [label_to_num[label] for label in predicted_trends]

    print("Model Performance Metrics:\n")
    print("Classification Report:")
    print(classification_report(known_trends, predicted_trends, zero_division=0))

    print("\nAdditional Metrics:")
    print(f"Accuracy: {accuracy_score(known_trends, predicted_trends):.4f}")
    print(f"Precision: {precision_score(known_trends, predicted_trends, average='binary', pos_label='Up', zero_division=0):.4f}")
    print(f"Recall: {recall_score(known_trends, predicted_trends, average='binary', pos_label='Up', zero_division=0):.4f}")
    print(f"F1-score: {f1_score(known_trends, predicted_trends, average='binary', pos_label='Up', zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(known_trends, predicted_trends, labels=unique_labels))

generate_and_evaluate_corrected_data()