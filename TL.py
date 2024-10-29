import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define the dataset class
class VisionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    self.image_paths.append(os.path.join(label_dir, image_name))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
train_dataset = VisionDataset(root_dir='/home/phd/dataset/VISION/train', transform=transform)
val_dataset = VisionDataset(root_dir='/home/phd/dataset/VISION/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the models to be trained
models_dict = {
    'MobileNet': models.mobilenet_v2(pretrained=True),
    'EfficientNet': models.efficientnet_b0(pretrained=True),
    'TinyNet': models.mnasnet1_0(pretrained=True),
    'ResNet101': models.resnet101(pretrained=True),
    'Xception': models.xception(pretrained=True),
    'Inception': models.inception_v3(pretrained=True),
    'UNet': models.segmentation.unet(pretrained=True),
    'VNet': models.segmentation.vnet(pretrained=True),
    'UNet++': models.segmentation.unetplusplus(pretrained=True)
}

# Modify the final layer of each model to match the number of classes in the dataset
num_classes = len(os.listdir('/home/phd/dataset/VISION/train'))
for model_name in models_dict:
    model = models_dict[model_name]
    if model_name in ['Inception']:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ['UNet', 'VNet', 'UNet++']:
        model.classifier[4] = nn.Conv2d(model.classifier[4].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc)

        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

    return train_acc_history, val_acc_history, train_loss_history, val_loss_history

# Define the visualization function
def visualize_results(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
    epochs_range = range(len(train_acc_history))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_history, label='Training Accuracy')
    plt.plot(epochs_range, val_acc_history, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_history, label='Training Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

# Define the function to plot confusion matrix and ROC curve
def plot_confusion_matrix_and_roc(model, val_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    plt.show()

# Training and evaluating the models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

for model_name, model in models_dict.items():
    print(f'Training {model_name}...')
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_acc_history, val_acc_history, train_loss_history, val_loss_history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)
    visualize_results(train_acc_history, val_acc_history, train_loss_history, val_loss_history)
    plot_confusion_matrix_and_roc(model, val_loader)
