import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment

        # Paths
        self.points_dir = os.path.join(root_dir, split, 'points')
        self.labels_dir = os.path.join(root_dir, split, 'labels')

        # File list
        self.file_list = sorted(os.listdir(self.points_dir))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # File name
        file_name = self.file_list[idx]

        # Load .npy files
        point_cloud = np.load(os.path.join(self.points_dir, file_name))  # (N, 3)
        label = np.load(os.path.join(self.labels_dir, file_name))        # int

        # Optional: data augmentation
        if self.augment:
            point_cloud = self.random_rotate(point_cloud)
            point_cloud = self.random_jitter(point_cloud)

        # Convert to torch tensors
        point_cloud = torch.from_numpy(point_cloud).float()  # (N, 3)
        label = torch.tensor(label).long()                   # scalar
        print("point_cloud shape:", point_cloud.shape)
        return point_cloud, label

    def random_rotate(self, pc):
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        return np.dot(pc, rotation_matrix)

    def random_jitter(self, pc, sigma=0.01, clip=0.05):
        jitter = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
        return pc + jitter
