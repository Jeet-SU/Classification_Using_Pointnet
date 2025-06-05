import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset.dataset import ModelNet40Dataset
from Model.pointnet_cls import PointNetCls

# ------------------------ CONFIG ------------------------
NUM_CLASSES = 6
NUM_POINTS = 1024
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Model/best_model.pth"

# ------------------------ LOAD MODEL ------------------------
model = PointNetCls(k=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------------ LOAD TEST DATA ------------------------
test_dataset = ModelNet40Dataset(
    root_dir='/home/netramn/Desktop/PointNet Classification/POINTNET_40/dataset/processed_modelnet40_fps',
    split='test',
    num_points=NUM_POINTS,
    augment=False
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ------------------------ EVALUATE ------------------------
all_preds = []
all_labels = []
misclassified = []

with torch.no_grad():
    for batch_idx, (points, labels) in enumerate(test_loader):
        points, labels = points.to(DEVICE), labels.to(DEVICE)
        # points = points.transpose(2, 1)  # B x 3 x N
        outputs, _ = model(points)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(points.size(0)):
            if preds[i] != labels[i]:
                misclassified.append({
                    "index": batch_idx * BATCH_SIZE + i,
                    "true_label": int(labels[i].item()),
                    "predicted_label": int(preds[i].item())
                })

# ------------------------ METRICS ------------------------
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = np.mean(all_preds == all_labels)
print(f"\n Final Test Accuracy: {acc*100:.2f}%")

print("\n Classification Report:\n")
print(classification_report(all_labels, all_preds, digits=4))

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
ious = []
for i in range(NUM_CLASSES):
    TP = conf_matrix[i, i]
    FP = np.sum(conf_matrix[:, i]) - TP
    FN = np.sum(conf_matrix[i, :]) - TP
    iou = TP / (TP + FP + FN + 1e-6)
    ious.append(iou)

mean_iou = np.mean(ious)

# ------------------------ SAVE METRICS ------------------------
metrics = {
    "final_test_accuracy": acc,
    "per_class_iou": ious,
    "mean_iou": mean_iou,
    "num_test_samples": len(test_dataset),
    "num_misclassified": len(misclassified)
}

with open("test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(" Metrics saved to `test_metrics.json`")

# ------------------------ SAVE PREDICTIONS ------------------------
with open("test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Sample", "True Label", "Predicted Label"])
    for i, (t, p) in enumerate(zip(all_labels, all_preds)):
        writer.writerow([i, t, p])
print(" Predictions saved to `test_predictions.csv`")

# # ------------------------ SAVE MISCLASSIFIED ------------------------
# with open("misclassified_samples.csv", "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=["index", "true_label", "predicted_label"])
#     writer.writeheader()
#     writer.writerows(misclassified)
# print(" Misclassified samples saved to `misclassified_samples.csv`")
# ------------------------ SAVE ALL CLASSIFICATIONS ------------------------
classified_all = []
for idx, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
    classified_all.append({
        "index": idx,
        "true_label": int(true_label),
        "predicted_label": int(pred_label)
    })

with open("classified_all.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "true_label", "predicted_label"])
    writer.writeheader()
    writer.writerows(classified_all)
print(" All classification results saved to `classified_all.csv`")

# ------------------------ CONFUSION MATRIX PLOT ------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("test_confusion_matrix.png")
plt.show()
print("Confusion matrix saved to `test_confusion_matrix.png`")
