import os
import numpy as np
import glob
import random
from tqdm import tqdm

# Settings
DATA_ROOT = "/home/netramn/Desktop/PointNet Classification/POINTNET_40/ModelNet40/ModelNet40"
OUTPUT_DIR = "processed_modelnet40_rs"
NUM_POINTS = 1024
SPLITS = ['train', 'test']

# # OFF parser
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


# Normalize and center
def normalize_point_cloud(pc):
    pc = pc - np.mean(pc, axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / scale
    return pc

# Uniform random sampling
def sample_point_cloud(pc, num_points):
    if pc.shape[0] >= num_points:
        indices = np.random.choice(pc.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(pc.shape[0], num_points, replace=True)
    return pc[indices]

# Get all classes
classes = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
class_to_idx = {cls: i for i, cls in enumerate(classes)}

# Create output directories
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "points"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Process data
for cls in tqdm(classes, desc="Processing classes"):
    label = class_to_idx[cls]
    for split in SPLITS:
        folder = os.path.join(DATA_ROOT, cls, split)
        files = glob.glob(os.path.join(folder, "*.off"))
        for filepath in files:
            try:
                pc = load_off_file(filepath)
                pc = normalize_point_cloud(pc)
                pc = sample_point_cloud(pc, NUM_POINTS)
                basename = os.path.splitext(os.path.basename(filepath))[0]
                save_name = f"{cls}_{basename}.npy"

                # Save points and label separately
                np.save(os.path.join(OUTPUT_DIR, split, "points", save_name), pc)
                np.save(os.path.join(OUTPUT_DIR, split, "labels", save_name), label)
            except Exception as e:
                print(f"Failed to process {filepath}: {e}")

# Save label mapping
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    for cls in classes:
        f.write(f"{class_to_idx[cls]},{cls}\n")

print("✅ Preprocessing complete!")
