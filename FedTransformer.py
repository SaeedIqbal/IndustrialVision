import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import numpy as np

# Parameters for federated learning
NUM_CLIENTS = 5  # Example number of nodes in federated learning
CLIENT_DATA_SIZES = [100, 150, 120, 130, 140]  # Each node's data size

# Phase 1: Vision Transformer (ViT) Model for Defect Classification
class ViTDefectClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ViTDefectClassifier, self).__init__()
        
        # Initialize Vision Transformer model (ViT) for image feature extraction
        config = ViTConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, 
                           image_size=224, patch_size=16, num_channels=3, attention_probs_dropout_prob=0.1)
        self.vit = ViTModel(config)
        
        # Linear layer for defect classification
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        vit_outputs = self.vit(x).last_hidden_state  # Extract features from ViT
        cls_token = vit_outputs[:, 0]  # Extract the [CLS] token
        logits = self.classifier(cls_token)
        return logits

# Define Cross-Entropy Loss for defect classification
criterion = nn.CrossEntropyLoss()

# Example setup for federated learning training
class Client:
    def __init__(self, model, data_size):
        self.model = model
        self.data_size = data_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train_on_local_data(self, data_loader):
        self.model.train()
        total_loss = 0
        for images, labels in data_loader:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

# Phase 2: Federated Learning with Weighted Aggregation
def federated_weighted_aggregation(clients):
    global_model = ViTDefectClassifier(num_classes=10)
    global_weights = {name: torch.zeros_like(param) for name, param in global_model.state_dict().items()}
    total_samples = sum([client.data_size for client in clients])
    
    # Aggregate weights from each client using weighted averaging
    with torch.no_grad():
        for client in clients:
            client_weight = client.data_size / total_samples
            for name, param in client.model.state_dict().items():
                global_weights[name] += param * client_weight

        # Update global model with aggregated weights
        global_model.load_state_dict(global_weights)
    
    return global_model

# Dummy data loading function for demonstration purposes
def get_dataloader(data_size, batch_size=16):
    # Placeholder for actual data loading logic
    dataset = [torch.rand(3, 224, 224), torch.randint(0, 10, (1,)) for _ in range(data_size)]
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize clients for federated learning
clients = [Client(ViTDefectClassifier(num_classes=10), data_size) for data_size in CLIENT_DATA_SIZES]

# Federated Learning Training Loop
global_model = ViTDefectClassifier(num_classes=10)
for round in range(5):  # Number of federated rounds
    print(f"Federated Training Round {round+1}")
    
    # Local training on each client
    for i, client in enumerate(clients):
        dataloader = get_dataloader(client.data_size)
        local_loss = client.train_on_local_data(dataloader)
        print(f"Client {i+1}, Local Loss: {local_loss:.4f}")

    # Aggregate weights using weighted average rule
    global_model = federated_weighted_aggregation(clients)
    
    # Distribute updated global weights to each client
    for client in clients:
        client.model.load_state_dict(global_model.state_dict())

print("Federated training complete.")
