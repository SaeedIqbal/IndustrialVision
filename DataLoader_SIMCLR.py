import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt

# Define the dataset classes
classes = ['Cable', 'Capacitor', 'Casting', 'Console', 'Cylinder', 'Electronics', 'Groove', 'Hemisphere', 'Lens', 'PCB_1', 'PCB_2', 'Ring', 'Screw', 'Wood']

# Phase 1: Data Preparation with Data Augmentation and Preprocessing
# Define data augmentation and preprocessing transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the VISION dataset (assuming data is organized in folders)
train_data = datasets.ImageFolder(root='/home/phd/dataset/VISION/train', transform=data_transform)
test_data = datasets.ImageFolder(root='/home/phd/dataset/VISION/train', transform=data_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Function to show images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get a batch of training data for visualization
dataiter = iter(train_loader)
images, labels = dataiter.next()

# Show images before augmentation and preprocessing
print('Images before augmentation and preprocessing:')
imshow(torchvision.utils.make_grid(images))

# Phase 2: Advanced Self-Supervised Learning using Contrastive Learning
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder):
        super(SimCLRModel, self).__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        z = self.projection_head(h)
        return h, z

class MoCoModel(nn.Module):
    def __init__(self, base_encoder, momentum=0.999):
        super(MoCoModel, self).__init__()
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder
        self.projection_head_q = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.projection_head_k = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.momentum = momentum
        self._initialize_momentum_encoder()

    def _initialize_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def forward(self, x_q, x_k):
        q = self.encoder_q(x_q)
        q = torch.flatten(q, start_dim=1)
        q = self.projection_head_q(q)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)
            k = torch.flatten(k, start_dim=1)
            k = self.projection_head_k(k)

        return q, k

    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

base_encoder = models.resnet50(pretrained=False)
simclr_model = SimCLRModel(base_encoder)
moco_model = MoCoModel(base_encoder)

# Contrastive Loss for SimCLR and MoCo
def contrastive_loss(features, temperature=0.5):
    batch_size = features.size(0)
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    loss = -torch.log(torch.sum(labels * F.softmax(similarity_matrix, dim=1), dim=1)).mean()
    return loss

# Example Training Execution for SimCLR
optimizer_simclr = optim.Adam(simclr_model.parameters(), lr=1e-3)
train_model(simclr_model, train_loader, contrastive_loss, optimizer_simclr, num_epochs=5)

# Example Training Execution for MoCo
optimizer_moco = optim.Adam(moco_model.parameters(), lr=1e-3)
train_model(moco_model, train_loader, contrastive_loss, optimizer_moco, num_epochs=5)

# Show images after augmentation and preprocessing for comparison
print('Images after augmentation and preprocessing:')
imshow(torchvision.utils.make_grid(images))
