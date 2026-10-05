"""Evaluate the saved EfficientNetB2 model and create score visuals."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from train_efficientnetb2_local import CLASS_NAMES, MODEL_DIR, build_model, load_dataset


ROOT = Path(__file__).resolve().parents[1]
VISUAL_DIR = MODEL_DIR / "visuals"

x, y = load_dataset()
_, x_test, _, y_test = train_test_split(
    x, y, test_size=0.2, random_state=101, stratify=y
)

model = tf.keras.models.load_model(MODEL_DIR / "efficientnetb2_local.keras")
probabilities = model.predict(x_test, batch_size=32, verbose=1)
predicted = np.argmax(probabilities, axis=1)
report = classification_report(
    y_test, predicted, target_names=CLASS_NAMES, output_dict=True
)

VISUAL_DIR.mkdir(parents=True, exist_ok=True)
(VISUAL_DIR / "classification_report.txt").write_text(
    classification_report(y_test, predicted, target_names=CLASS_NAMES, digits=4)
)

matrix = confusion_matrix(y_test, predicted)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
    ax=ax, cmap="Blues", values_format="d", colorbar=False
)
ax.set_title("EfficientNetB2 Test Confusion Matrix")
fig.tight_layout()
fig.savefig(VISUAL_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

metrics = ["precision", "recall", "f1-score"]
values = np.array([[report[name][metric] for metric in metrics] for name in CLASS_NAMES])
fig, ax = plt.subplots(figsize=(8, 5))
x_positions = np.arange(len(CLASS_NAMES))
width = 0.24
for index, metric in enumerate(metrics):
    ax.bar(x_positions + (index - 1) * width, values[:, index], width, label=metric)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("EfficientNetB2 Per-Class Scores")
ax.set_xticks(x_positions, CLASS_NAMES)
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(VISUAL_DIR / "per_class_scores.png", dpi=160)
plt.close(fig)

summary_names = ["Accuracy", "Macro F1", "Weighted F1"]
summary_values = [
    report["accuracy"],
    report["macro avg"]["f1-score"],
    report["weighted avg"]["f1-score"],
]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(summary_names, summary_values, color=["#2563eb", "#0891b2", "#16a34a"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("EfficientNetB2 Overall Test Scores")
ax.grid(axis="y", alpha=0.25)
for bar, value in zip(bars, summary_values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center")
fig.tight_layout()
fig.savefig(VISUAL_DIR / "overall_scores.png", dpi=160)
plt.close(fig)

print(classification_report(y_test, predicted, target_names=CLASS_NAMES, digits=4))
print(f"Visuals saved to {VISUAL_DIR}")
