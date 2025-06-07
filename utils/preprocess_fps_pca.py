import os
import numpy as np
import glob
from tqdm import tqdm

# Configuration
DATA_ROOT = "/home/netramn/Desktop/PointNet Classification/POINTNET_40/ModelNet40/ModelNet40"
OUTPUT_DIR = "processed_modelnet40_fps"
NUM_POINTS = 1024
SPLITS = ['train', 'test']

# OFF parser
# def load_off_file(filepath):
#     with open(filepath, 'r') as f:
#         if f.readline().strip() != 'OFF':
#             raise ValueError('Not a valid OFF header')
#         counts = f.readline().strip().split()
#         num_vertices = int(counts[0])
#         vertices = []
#         for _ in range(num_vertices):
#             point = list(map(float, f.readline().strip().split()))
#             vertices.append(point)
#         return np.array(vertices, dtype=np.float32)
def load_off_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('OFF'):
            if len(first_line.split()) == 1:
                # Standard OFF: read counts on second line
                counts = f.readline().strip().split()
            else:
                # Compact OFF with counts on same line
                counts = first_line[3:].strip().split()
        else:
            raise ValueError("Not a valid OFF header")
        
        num_vertices = int(counts[0])
        vertices = []
        for _ in range(num_vertices):
            point = list(map(float, f.readline().strip().split()))
            vertices.append(point)
        return np.array(vertices, dtype=np.float32)
# Normalize to unit sphere
def normalize_point_cloud(pc):
    pc = pc - np.mean(pc, axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    return pc / scale

# Farthest Point Sampling (NumPy)
def farthest_point_sampling(points, num_samples):
    N, D = points.shape
    sampled_pts = np.zeros((num_samples, D), dtype=np.float32)
    distances = np.ones(N) * 1e10
    farthest_idx = np.random.randint(0, N)
    for i in range(num_samples):
        sampled_pts[i] = points[farthest_idx]
        dist = np.sum((points - points[farthest_idx]) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest_idx = np.argmax(distances)
    return sampled_pts
# Principal Componenet Analysis
def align_with_pca(pc):
    """
    Align the point cloud using PCA.
    Makes the main axes of the object aligned with X, Y, Z.
    """
    # Subtract mean
    pc = pc - np.mean(pc, axis=0)

    # Compute covariance matrix
    cov = np.cov(pc.T)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvectors by descending eigenvalues
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sort_idx]

    # Align points
    aligned_pc = np.dot(pc, eigvecs)
    return aligned_pc

# Class indexing
classes = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
class_to_idx = {cls: i for i, cls in enumerate(classes)}

# Output directories
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "points"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Processing
for cls in tqdm(classes, desc="Processing with FPS"):
    label = class_to_idx[cls]
    for split in SPLITS:
        folder = os.path.join(DATA_ROOT, cls, split)
        files = glob.glob(os.path.join(folder, "*.off"))
        for filepath in files:
            try:
                pc = load_off_file(filepath)
                pc = align_with_pca(pc)
                pc = farthest_point_sampling(pc, NUM_POINTS)
                pc = normalize_point_cloud(pc)   # Normalize after sampling
                basename = os.path.splitext(os.path.basename(filepath))[0]
                save_name = f"{cls}_{basename}.npy"

                np.save(os.path.join(OUTPUT_DIR, split, "points", save_name), pc)
                np.save(os.path.join(OUTPUT_DIR, split, "labels", save_name), label)
            except Exception as e:
                print(f"Failed: {filepath} due to {e}")

# Save class label map
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    for cls in classes:
        f.write(f"{class_to_idx[cls]},{cls}\n")

        files = glob.glob(os.path.join(folder, "*.off"))
        print(f"[{cls}/{split}] Found {len(files)} OFF files")


print(" PCA/FPS-based preprocessing complete!")
