# import os
# import time
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from tqdm import tqdm
# from collections import defaultdict
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader

# from dataset.dataset import ModelNet40Dataset
# from Model.pointnet_cls import PointNetCls, feature_transform_regularizer

# # ------------------------ CONFIG ------------------------
# NUM_CLASSES = 6
# NUM_POINTS = 1024
# BATCH_SIZE = 16
# EPOCHS = 30
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ------------------------ DATASET -----------------------
# train_dataset = ModelNet40Dataset(
#     root_dir='/home/netramn/Desktop/PointNet Classification/POINTNET_40/dataset/processed_modelnet40_fps',
#     split='train',
#     num_points=NUM_POINTS,
#     augment=True
# )

# test_dataset = ModelNet40Dataset(
#     root_dir='/home/netramn/Desktop/PointNet Classification/POINTNET_40/dataset/processed_modelnet40_fps',
#     split='test',
#     num_points=NUM_POINTS,
#     augment=False
# )

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# # ------------------------ MODEL -------------------------
# model = PointNetCls(k=NUM_CLASSES).to(DEVICE)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # ------------------ METRIC TRACKING ---------------------
# metrics = defaultdict(list)
# start_time = time.time()

# # --------------------- TRAINING LOOP --------------------
# for epoch in range(EPOCHS):
#     epoch_start = time.time()
#     model.train()
#     total_loss = 0
#     correct = total = 0

#     for points, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]"):
#         points, labels = points.to(DEVICE), labels.to(DEVICE)

#         outputs, trans_feat = model(points)

#         optimizer.zero_grad()
#         loss = criterion(outputs, labels)
#         loss += feature_transform_regularizer(trans_feat) * 0.001  # Optional regularizer
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         preds = outputs.argmax(dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#     acc = correct / total
#     epoch_time = time.time() - epoch_start

#     metrics['epoch'].append(epoch + 1)
#     metrics['loss'].append(total_loss)
#     metrics['accuracy'].append(acc)
#     metrics['epoch_time'].append(epoch_time)

#     print(f" Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc*100:.2f}%, Time={epoch_time:.2f}s")

# # -------------------- TEST EVALUATION -------------------
# model.eval()
# correct = total = 0
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for points, labels in test_loader:
#         points, labels = points.to(DEVICE), labels.to(DEVICE)
#         outputs, _ = model(points)

#         preds = outputs.argmax(dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

# test_acc = correct / total
# print(f"\n Final Test Accuracy: {test_acc*100:.2f}%")

# # ------------------ CONFUSION MATRIX & IOU ----------------
# conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
# for pred, label in zip(all_preds, all_labels):
#     conf_matrix[label][pred] += 1

# ious = []
# for i in range(NUM_CLASSES):
#     TP = conf_matrix[i, i]
#     FP = np.sum(conf_matrix[:, i]) - TP
#     FN = np.sum(conf_matrix[i, :]) - TP
#     iou = TP / (TP + FP + FN + 1e-6)
#     ious.append(iou)

# mean_iou = np.mean(ious)
# print(f"\n Per-class IoU : {[f'{iou*100:.2f}%' for iou in ious]}")
# print(f" Mean IoU      : {mean_iou*100:.2f}%")

# # -------------------- SUMMARY METRICS --------------------
# total_training_time = time.time() - start_time
# metrics['total_training_time'] = total_training_time
# metrics['total_objects'] = len(train_dataset) * EPOCHS
# metrics['final_test_accuracy'] = test_acc
# metrics['per_class_iou'] = ious
# metrics['mean_iou'] = mean_iou

# # -------------------- SAVE METRICS -----------------------
# with open("training_metrics.json", "w") as f:
#     json.dump(metrics, f, indent=2)

# # --------------------- PLOT METRICS ----------------------
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(metrics['epoch'], metrics['accuracy'], label='Accuracy')
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training Accuracy")
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(metrics['epoch'], metrics['loss'], label='Loss', color='red')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("training_graph.png")
# plt.show()

# print(f"\n Training complete. Metrics saved to `training_metrics.json`, plot saved to `training_graph.png`")

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset.dataset import ModelNet40Dataset
from Model.pointnet_cls import PointNetCls, feature_transform_regularizer

# ------------------------ CONFIG ------------------------
NUM_CLASSES = 6
NUM_POINTS = 1024
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ DATASET -----------------------
train_dataset = ModelNet40Dataset(
    root_dir='/home/netramn/Desktop/PointNet Classification/POINTNET_40/dataset/processed_modelnet40_fps',
    split='train',
    num_points=NUM_POINTS,
    augment=True
)

test_dataset = ModelNet40Dataset(
    root_dir='/home/netramn/Desktop/PointNet Classification/POINTNET_40/dataset/processed_modelnet40_fps',
    split='test',
    num_points=NUM_POINTS,
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ------------------------ MODEL -------------------------
model = PointNetCls(k=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------ METRIC TRACKING ---------------------
metrics = defaultdict(list)
start_time = time.time()
best_acc = 0.0
os.makedirs("models", exist_ok=True)

# --------------------- TRAINING LOOP --------------------
for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    correct = total = 0

    for points, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]"):
        points, labels = points.to(DEVICE), labels.to(DEVICE)

        outputs, trans_feat = model(points)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    epoch_time = time.time() - epoch_start

    metrics['epoch'].append(epoch + 1)
    metrics['loss'].append(total_loss)
    metrics['accuracy'].append(acc)
    metrics['epoch_time'].append(epoch_time)

    print(f" Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc*100:.2f}%, Time={epoch_time:.2f}s")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "Model/best_model.pth")
        print(f"  Best model saved at epoch {epoch+1} with accuracy {acc*100:.2f}%")

# -------------------- TEST EVALUATION -------------------
model.eval()
correct = total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for points, labels in test_loader:
        points, labels = points.to(DEVICE), labels.to(DEVICE)
        outputs, _ = model(points)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\n  Final Test Accuracy: {test_acc*100:.2f}%")

# ------------------ CONFUSION MATRIX & IOU ----------------
conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
for pred, label in zip(all_preds, all_labels):
    conf_matrix[label][pred] += 1

ious = []
for i in range(NUM_CLASSES):
    TP = conf_matrix[i, i]
    FP = np.sum(conf_matrix[:, i]) - TP
    FN = np.sum(conf_matrix[i, :]) - TP
    iou = TP / (TP + FP + FN + 1e-6)
    ious.append(iou)

mean_iou = np.mean(ious)
print(f"\n  Per-class IoU : {[f'{iou*100:.2f}%' for iou in ious]}")
print(f"  Mean IoU      : {mean_iou*100:.2f}%")

# -------------------- SUMMARY METRICS --------------------
total_training_time = time.time() - start_time
metrics['total_training_time'] = total_training_time
metrics['total_objects'] = len(train_dataset) * EPOCHS
metrics['final_test_accuracy'] = test_acc
metrics['per_class_iou'] = ious
metrics['mean_iou'] = mean_iou

# -------------------- SAVE METRICS -----------------------
with open("training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# --------------------- PLOT METRICS ----------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(metrics['epoch'], metrics['accuracy'], label='Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(metrics['epoch'], metrics['loss'], label='Loss', color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_graph.png")
plt.show()

print("\n Training complete.")
print("Best model saved as `Model/best_model.pth`")
print("Metrics saved to `training_metrics.json`")
print(" Training plot saved to `training_graph.png`")
