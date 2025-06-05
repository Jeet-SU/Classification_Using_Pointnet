# import open3d as o3d
# import numpy as np

# def visualize_point_cloud_open3d(points, true_label, pred_label, class_names=None):
#     """
#     Visualizes a point cloud with Open3D.
#     Args:
#         points: numpy array (N,3) of XYZ coordinates
#         true_label: int, ground truth class
#         pred_label: int, predicted class
#         class_names: optional list of class names (strings)
#     """

#     # Create Open3D PointCloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # Color code: green if correct, red if misclassified
#     if true_label == pred_label:
#         colors = np.tile(np.array([[0, 1, 0]]), (points.shape[0], 1))  # green
#     else:
#         colors = np.tile(np.array([[1, 0, 0]]), (points.shape[0], 1))  # red
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Create visualizer window
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name="PointNet Classification Visualization")

#     vis.add_geometry(pcd)

#     # Add text annotation as window title
#     if class_names:
#         title = f"True: {class_names[true_label]} | Predicted: {class_names[pred_label]}"
#     else:
#         title = f"True: {true_label} | Predicted: {pred_label}"

#     vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])  # dark bg

#     vis.get_view_control().set_zoom(0.8)

#     # Set window title to labels
#     vis.get_window().setWindowTitle(title)

#     vis.run()
#     vis.destroy_window()

# # Example usage:

# # Suppose test_dataset is your dataset, all_preds and all_labels from test
# sample_idx = 0  # index of sample to visualize

# points_sample, label_sample = test_dataset[sample_idx]  # (N, 3), label
# pred_label_sample = all_preds[sample_idx]

# # Convert to numpy if tensor
# if hasattr(points_sample, "numpy"):
#     points_sample = points_sample.numpy()

# visualize_point_cloud_open3d(points_sample, label_sample, pred_label_sample)
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.modelnet40_dataset import ModelNet40Dataset  # adjust if your Dataset class is elsewhere

# Load test dataset
test_dataset = ModelNet40Dataset(
    root='/home/netramn/Desktop/PointNet Classification/POINTNET_40/dataset/processed_modelnet40_fps',
    split='test'
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Assuming these were generated earlier
# all_preds = [...]
# all_labels = [...]
# misclassified = [...]  # List of dicts: { "index": idx, "true_label": x, "predicted_label": y }

# Optional: label names
class_names = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
    "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
    "mantel", "monitor", "night_stand", "person", "piano", "plant",
    "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
    "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
]

def visualize_point_cloud(points_np, gt_label, pred_label):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Color by prediction: green = correct, red = incorrect
    if gt_label == pred_label:
        color = np.tile(np.array([[0, 1, 0]]), (points_np.shape[0], 1))  # green
    else:
        color = np.tile(np.array([[1, 0, 0]]), (points_np.shape[0], 1))  # red
    pcd.colors = o3d.utility.Vector3dVector(color)

    # Viewer
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"GT: {class_names[gt_label]} | Pred: {class_names[pred_label]}"
    )

# === Loop Through Misclassified Samples ===
for entry in misclassified:
    idx = entry["index"]
    gt = entry["true_label"]
    pred = entry["predicted_label"]

    points, _ = test_dataset[idx]
    points_np = points.numpy()  # [N, 3]

    visualize_point_cloud(points_np, gt, pred)
