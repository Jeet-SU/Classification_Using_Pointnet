import torch
import torch.nn as nn
import torch.nn.functional as F

# T-Net (Spatial Transformer Network)
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))   # B x 64 x N
        x = F.relu(self.bn2(self.conv2(x)))   # B x 128 x N
        x = F.relu(self.bn3(self.conv3(x)))   # B x 1024 x N

        x = torch.max(x, 2, keepdim=False)[0] # B x 1024

        x = F.relu(self.bn4(self.fc1(x)))     # B x 512
        x = F.relu(self.bn5(self.fc2(x)))     # B x 256
        x = self.fc3(x)                       # B x (k*k)

        # Initialize as identity matrix
        identity = torch.eye(self.k, requires_grad=True).repeat(B, 1, 1).to(x.device)
        x = x.view(-1, self.k, self.k) + identity
        return x


# PointNet Classification Network
class PointNetCls(nn.Module):
    def __init__(self, k=40):  # k = number of output classes
        super(PointNetCls, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(256, k)

    def forward(self, x):  # x: B x N x 3
        B, N, _ = x.size()
        x = x.transpose(2, 1)  # B x 3 x N

        # Input Transform
        trans = self.input_transform(x)             # B x 3 x 3
        x = torch.bmm(trans, x)                     # B x 3 x N

        # MLP(64)
        x = F.relu(self.bn1(self.conv1(x)))         # B x 64 x N

        # Feature Transform
        trans_feat = self.feature_transform(x)      # B x 64 x 64
        x = torch.bmm(trans_feat, x)                # B x 64 x N

        # MLP(128, 1024)
        x = F.relu(self.bn2(self.conv2(x)))         # B x 128 x N
        x = self.bn3(self.conv3(x))                 # B x 1024 x N

        # Global feature vector
        x = torch.max(x, 2)[0]                      # B x 1024

        # Fully Connected Layers
        x = F.relu(self.bn4(self.fc1(x)))           # B x 512
        x = self.drop1(x)
        x = F.relu(self.bn5(self.fc2(x)))           # B x 256
        x = self.drop2(x)
        x = self.fc3(x)                             # B x k

        return x, trans_feat  # return transformation matrix for feature transform regularization
def feature_transform_regularizer(trans):
    """
    Computes the regularization loss for the feature transform matrix.
    Encourages the matrix to be orthogonal: T * T^T ≈ I
    """
    d = trans.size(1)
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
